import cv2 as cv
import numpy as np
import math
import itertools

# --- CONFIGURATION ---
CORNER_CLUSTER_DIST = 50  # Max dist to group dots into a corner
MIN_CARD_AREA = 10000  # Min area to be a valid card
MAX_CARD_AREA = 120000  # Max area (prevent finding the whole table)
ASPECT_RATIO_TOLERANCE = 0.4  # How much it can deviate from a square/rectangle


def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def split_double_corner(points):
    """
    If 6 dots are found together, it's likely two corners touching.
    We split them based on the widest axis (horizontal or vertical).
    """
    pts = np.array(points)
    # Find bounding box variance
    x_var = np.var(pts[:, 0])
    y_var = np.var(pts[:, 1])

    # Sort and split based on the spread
    if x_var > y_var:
        pts = pts[pts[:, 0].argsort()]  # Sort by X
    else:
        pts = pts[pts[:, 1].argsort()]  # Sort by Y

    # Split into two groups of 3
    group1 = pts[:3]
    group2 = pts[3:]

    c1 = (int(np.mean(group1[:, 0])), int(np.mean(group1[:, 1])))
    c2 = (int(np.mean(group2[:, 0])), int(np.mean(group2[:, 1])))
    return [c1, c2]


def order_points(pts):
    # Sort points to: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def is_valid_rectangle(corners):
    """
    Checks if 4 points form a valid rectangle.
    """
    pts = np.array(corners, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 1. Check Dimensions
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    max_w = max(int(widthA), int(widthB))
    max_h = max(int(heightA), int(heightB))

    # Area Check
    area = max_w * max_h
    if area < MIN_CARD_AREA or area > MAX_CARD_AREA:
        return False, None

    # 2. Check Parallelism / Aspect Ratio consistency
    # (Simple check: Opposite sides should be roughly equal length)
    if abs(widthA - widthB) > 30 or abs(heightA - heightB) > 30:
        return False, None

    return True, rect


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Multi-Card Detection Started.")
    print("Looking for groups of 4 corners...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # 1. Find raw dots
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
                    all_dots.append([cx, cy])

        # 2. Cluster dots into Corners
        # Simple distance grouping
        corners = []
        processed = [False] * len(all_dots)

        for i in range(len(all_dots)):
            if processed[i]: continue

            cluster = [all_dots[i]]
            processed[i] = True

            # Find neighbors
            for j in range(i + 1, len(all_dots)):
                if not processed[j]:
                    dist = get_dist(all_dots[i], all_dots[j])
                    if dist < CORNER_CLUSTER_DIST:
                        cluster.append(all_dots[j])
                        processed[j] = True

            # Analyze Cluster
            # Exact corner = 3 dots
            if len(cluster) == 3:
                avg_x = int(sum(p[0] for p in cluster) / 3)
                avg_y = int(sum(p[1] for p in cluster) / 3)
                corners.append((avg_x, avg_y))
                cv.circle(frame, (avg_x, avg_y), 10, (255, 255, 0), 2)  # Blue Circle = Corner

            # Double corner (Touching cards) = 6 dots
            elif len(cluster) >= 5 and len(cluster) <= 7:
                split_corners = split_double_corner(cluster)
                corners.extend(split_corners)
                for sc in split_corners:
                    cv.circle(frame, sc, 10, (0, 255, 255), 2)  # Yellow Circle = Split Corner

        # 3. Find Rectangles (Combinations of 4 corners)
        # Only run if we have enough corners for at least one card
        if len(corners) >= 4:
            # If we have too many corners (e.g. noise), limit checks to prevent lag
            # 12 corners = 495 combinations (Fast). 20 corners = 4845 (Okay).
            if len(corners) > 16:
                corners = corners[:16]

                # Check every combination of 4 corners
            for quad in itertools.combinations(corners, 4):
                is_valid, rect_points = is_valid_rectangle(quad)

                if is_valid:
                    # Draw the card boundary
                    box = np.int32(rect_points)
                    cv.drawContours(frame, [box], 0, (0, 255, 0), 3)

                    # Label
                    M = cv.moments(box)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv.putText(frame, "Card", (cx - 20, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Multi-Card Detection', frame)
        # cv.imshow('Thresh', thresh) # Uncomment to debug dot detection

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()