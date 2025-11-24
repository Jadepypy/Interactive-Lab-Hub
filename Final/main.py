import cv2 as cv
import numpy as np
import math
import itertools

# --- CONFIGURATION ---
CORNER_CLUSTER_DIST = 50  # Max dist to group dots into a corner
MIN_CARD_AREA = 10000  # Min area to be a valid card
MAX_CARD_AREA = 120000  # Max area (prevent finding the whole table)
ALIGNMENT_TOLERANCE = 0.90  # Dot product threshold (1.0 is perfect parallel)


def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_corner_alignment(cluster):
    """
    Analyzes a cluster of 3 dots to find the L-shape orientation.
    Returns the two unit vectors representing the corner's edges.
    """
    p0, p1, p2 = cluster
    # Calculate all 3 distances to find the hypotenuse
    d01 = get_dist(p0, p1)
    d12 = get_dist(p1, p2)
    d20 = get_dist(p2, p0)

    dists = [d01, d12, d20]
    max_d = max(dists)

    # The vertex is the point opposite the longest side (hypotenuse)
    if max_d == d01:
        vertex, end1, end2 = p2, p0, p1
    elif max_d == d12:
        vertex, end1, end2 = p0, p1, p2
    else:
        vertex, end1, end2 = p1, p2, p0

    # Calculate Unit Vectors from Vertex to the two Ends
    def get_vec(start, end):
        vx = end[0] - start[0]
        vy = end[1] - start[1]
        mag = math.hypot(vx, vy)
        if mag == 0: return (0, 0)
        return (vx / mag, vy / mag)

    v1 = get_vec(vertex, end1)
    v2 = get_vec(vertex, end2)

    return [v1, v2]


def check_alignment(corner_vectors, edge_vector):
    """
    Checks if the edge_vector is parallel to EITHER of the corner's own alignment vectors.
    """
    ex, ey = edge_vector
    # Normalize edge vector
    mag = math.hypot(ex, ey)
    if mag == 0: return False
    ex, ey = ex / mag, ey / mag

    for vx, vy in corner_vectors:
        # Dot product: If parallel, dot is 1.0 or -1.0
        dot = (ex * vx) + (ey * vy)
        if abs(dot) > ALIGNMENT_TOLERANCE:
            return True

    return False


def split_double_corner(points):
    """
    Splits a group of 6 dots into two groups of 3 based on spatial spread.
    """
    pts = np.array(points)
    x_var = np.var(pts[:, 0])
    y_var = np.var(pts[:, 1])

    if x_var > y_var:
        pts = pts[pts[:, 0].argsort()]
    else:
        pts = pts[pts[:, 1].argsort()]

    return [pts[:3].tolist(), pts[3:].tolist()]


def order_corners_with_data(corners):
    """
    Sorts corners (which include alignment data) into TL, TR, BR, BL order.
    """
    # Extract just coordinates for sorting
    pts = np.array([(c['x'], c['y']) for c in corners], dtype="float32")

    s = pts.sum(axis=1)
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = corners[np.argmin(diff)]
    bl = corners[np.argmax(diff)]

    return [tl, tr, br, bl]


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Strict Alignment Detection Started.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # 1. Find Raw Dots
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        all_dots = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 20 or area > 2000: continue

            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity > 0.6:
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    all_dots.append((cx, cy))

        # 2. Cluster Dots into Valid Corners
        valid_corners = []  # Will store dicts: {'x':_, 'y':_, 'vectors':_}
        processed = [False] * len(all_dots)

        for i in range(len(all_dots)):
            if processed[i]: continue

            cluster = [all_dots[i]]
            processed[i] = True

            for j in range(i + 1, len(all_dots)):
                if not processed[j]:
                    dist = get_dist(all_dots[i], all_dots[j])
                    if dist < CORNER_CLUSTER_DIST:
                        cluster.append(all_dots[j])
                        processed[j] = True

            # Process Clusters
            final_clusters = []

            if len(cluster) == 3:
                final_clusters.append(cluster)
            elif len(cluster) >= 5 and len(cluster) <= 7:
                # Detected two corners touching
                final_clusters.extend(split_double_corner(cluster))

            # Analyze each valid 3-dot cluster
            for c_pts in final_clusters:
                if len(c_pts) != 3: continue

                # Center of the corner
                avg_x = int(sum(p[0] for p in c_pts) / 3)
                avg_y = int(sum(p[1] for p in c_pts) / 3)

                # Get L-shape alignment
                vectors = get_corner_alignment(c_pts)

                valid_corners.append({
                    'x': avg_x,
                    'y': avg_y,
                    'vectors': vectors
                })

                # Debug: Draw Corner and its alignment lines
                cv.circle(frame, (avg_x, avg_y), 5, (255, 255, 0), -1)
                # Draw short red lines indicating the detected orientation
                end_x1 = int(avg_x + vectors[0][0] * 20)
                end_y1 = int(avg_y + vectors[0][1] * 20)
                cv.line(frame, (avg_x, avg_y), (end_x1, end_y1), (0, 0, 255), 2)

        # 3. Find Cards (Combinations of 4 Corners)
        if len(valid_corners) >= 4:
            # Limit to 16 corners to maintain FPS
            if len(valid_corners) > 16: valid_corners = valid_corners[:16]

            for quad in itertools.combinations(valid_corners, 4):
                # 3a. Geometric Sort (TL, TR, BR, BL)
                sorted_quad = order_corners_with_data(quad)
                tl, tr, br, bl = sorted_quad

                # 3b. ALIGNMENT CHECK (The New Filter)
                # Check Top Edge (TL -> TR)
                top_edge_vec = (tr['x'] - tl['x'], tr['y'] - tl['y'])
                if not check_alignment(tl['vectors'], top_edge_vec): continue

                # Check Left Edge (TL -> BL)
                left_edge_vec = (bl['x'] - tl['x'], bl['y'] - tl['y'])
                if not check_alignment(tl['vectors'], left_edge_vec): continue

                # If we passed those checks, the shape is aligned with the dots!
                # Now we do the standard rectangle/area check

                width = get_dist((tl['x'], tl['y']), (tr['x'], tr['y']))
                height = get_dist((tl['x'], tl['y']), (bl['x'], bl['y']))
                area = width * height

                if area < MIN_CARD_AREA or area > MAX_CARD_AREA: continue

                # Draw Valid Card
                box = np.array([
                    [tl['x'], tl['y']], [tr['x'], tr['y']],
                    [br['x'], br['y']], [bl['x'], bl['y']]
                ], dtype="int32")

                cv.drawContours(frame, [box], 0, (0, 255, 0), 3)

                # Label center
                cx = int((tl['x'] + br['x']) / 2)
                cy = int((tl['y'] + br['y']) / 2)
                cv.putText(frame, "ID Ready", (cx - 40, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow('Strict Alignment', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()