import cv2 as cv
import numpy as np
import math

# --- CONFIGURATION ---
DOT_SIZE_MIN = 20
DOT_SIZE_MAX = 2000
CLUSTER_DIST = 60  # Max distance between dots to form a corner
CARD_MIN_WIDTH = 50  # Min pixel width of a card
CARD_MAX_WIDTH = 600
CARD_MIN_HEIGHT = 50
CARD_MAX_HEIGHT = 600


def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_angle(v1, v2):
    """ Returns the angle between two vectors in degrees. """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0: return 0
    # Clip to handle floating point errors slightly outside -1,1
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def classify_corner_shape(cluster):
    """
    Takes 3 points.
    1. Identifies the Vertex (the dot with the 90-degree angle).
    2. Determines orientation (TL, TR, BR, BL) based on where the 'arms' point.
    Returns: (CornerType, VertexPoint) or (None, None)
    """
    p0, p1, p2 = cluster

    # Calculate distances to find hypotenuse (longest side)
    d01 = get_dist(p0, p1)
    d12 = get_dist(p1, p2)
    d20 = get_dist(p2, p0)

    # The vertex is opposite the longest side
    sides = [(d12, p0, p1, p2), (d20, p1, p0, p2), (d01, p2, p0, p1)]
    sides.sort(key=lambda x: x[0])  # Sort by length, hypotenuse last

    hypotenuse, vertex, arm1, arm2 = sides[2]

    # Check if it's roughly 90 degrees
    vec1 = (arm1[0] - vertex[0], arm1[1] - vertex[1])
    vec2 = (arm2[0] - vertex[0], arm2[1] - vertex[1])

    angle = get_angle(vec1, vec2)
    if angle < 70 or angle > 110:  # Allow some tolerance
        return None, None

    # Determine Orientation using the Vector Sum (Diagonal direction)
    # Note: Image Y is Positive DOWN
    diag_x = vec1[0] + vec2[0]
    diag_y = vec1[1] + vec2[1]

    # Simple axis-aligned logic (robust for rotation +/- 45 deg)
    corner_type = None

    if diag_x > 0 and diag_y > 0:
        corner_type = "TL"  # Pointing Right and Down -> Top Left Corner
    elif diag_x < 0 and diag_y > 0:
        corner_type = "TR"  # Pointing Left and Down -> Top Right Corner
    elif diag_x < 0 and diag_y < 0:
        corner_type = "BR"  # Pointing Left and Up -> Bottom Right Corner
    elif diag_x > 0 and diag_y < 0:
        corner_type = "BL"  # Pointing Right and Up -> Bottom Left Corner

    return corner_type, vertex


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Pattern Matching Started. Looking for TL, TR, BR, BL L-shapes...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # 1. Detect Raw Dots
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        all_dots = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < DOT_SIZE_MIN or area > DOT_SIZE_MAX: continue

            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity > 0.6:
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    all_dots.append((cx, cy))

        # 2. Identify L-Shapes (Corners)
        corners = {"TL": [], "TR": [], "BR": [], "BL": []}

        # Naive clustering: Find groups of 3
        # In a real scenario with many dots, KD-Tree is faster, but this is fine for ~50 dots
        used_dots = [False] * len(all_dots)

        for i in range(len(all_dots)):
            if used_dots[i]: continue

            # Find neighbors
            cluster = [all_dots[i]]
            for j in range(i + 1, len(all_dots)):
                if not used_dots[j] and get_dist(all_dots[i], all_dots[j]) < CLUSTER_DIST:
                    cluster.append(all_dots[j])

            # Check if it's a triad (L-Shape)
            if len(cluster) == 3:
                c_type, vertex = classify_corner_shape(cluster)
                if c_type:
                    corners[c_type].append(vertex)
                    # Mark dots as used so they aren't reused
                    # (In complex overlaps, we might want to check all perms, but this assumes distinct spacing)
                    # For now, just visualizing the detected corner vertex
                    color_map = {"TL": (0, 255, 0), "TR": (255, 0, 0), "BR": (0, 0, 255), "BL": (0, 255, 255)}
                    cv.circle(frame, vertex, 8, color_map[c_type], -1)
                    cv.putText(frame, c_type, (vertex[0] - 10, vertex[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color_map[c_type], 2)

        # 3. Match Corners to Form Rectangles
        # We iterate TLs and try to find matching TR, BL, BR
        used_corners = {"TL": set(), "TR": set(), "BR": set(), "BL": set()}

        for tl in corners["TL"]:
            best_card = None
            min_error = float('inf')

            # Find TR (Should be to the RIGHT of TL, similar Y)
            for tr in corners["TR"]:
                if tr in used_corners["TR"]: continue
                if tr[0] > tl[0] and abs(tr[1] - tl[1]) < 50:  # Check alignment

                    # Find BL (Should be BELOW TL, similar X)
                    for bl in corners["BL"]:
                        if bl in used_corners["BL"]: continue
                        if bl[1] > tl[1] and abs(bl[0] - tl[0]) < 50:

                            # Find BR (Should be RIGHT of BL and BELOW TR)
                            for br in corners["BR"]:
                                if br in used_corners["BR"]: continue

                                # Verify BR is roughly where expected
                                expected_x = tr[0]
                                expected_y = bl[1]
                                dist_error = math.hypot(br[0] - expected_x, br[1] - expected_y)

                                if dist_error < 50:  # Tolerance for "rectangular-ness"
                                    # Found a valid quad!
                                    # Draw it
                                    cv.rectangle(frame, tl, br, (0, 255, 0), 3)
                                    cv.line(frame, tl, br, (0, 255, 0), 1)
                                    cv.line(frame, tr, bl, (0, 255, 0), 1)

                                    # Mark as used
                                    used_corners["TR"].add(tr)
                                    used_corners["BL"].add(bl)
                                    used_corners["BR"].add(br)
                                    break  # Stop looking for BR
                            else:
                                continue  # Continue looking for BL
                            break  # Stop looking for BL
                    else:
                        continue  # Continue looking for TR
                    break  # Stop looking for TR

        cv.imshow('Pattern Matching', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()