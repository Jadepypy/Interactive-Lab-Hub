import cv2 as cv
import numpy as np
import math
import itertools

# --- DEFAULTS ---
# Adjust these sliders in the window when the program runs
DEF_BLOCK_SIZE = 11  # Thresholding detail (Must be odd!)
DEF_C = 2  # Thresholding bias
DEF_MIN_AREA = 50  # Ignore spots smaller than this
DEF_CIRCULARITY = 50  # 0-100 scale (50 = 0.5 circularity)
CLUSTER_DIST = 250  # Max distance to look for neighbors
ANGLE_TOLERANCE = 0.25


def nothing(x):
    pass


def get_angle_cosine(vertex, p1, p2):
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0: return 1.0
    return abs(dot / (mag1 * mag2))


def order_points_relative_to_anchor(anchor, neighbors):
    pts = [anchor] + list(neighbors)
    dists = [math.hypot(p[0] - anchor[0], p[1] - anchor[1]) for p in pts]
    br_idx = np.argmax(dists)
    br = pts[br_idx]

    remain = [p for i, p in enumerate(pts) if i != 0 and i != br_idx]
    if len(remain) != 2: return None

    vec_diag = (br[0] - anchor[0], br[1] - anchor[1])
    vec_r0 = (remain[0][0] - anchor[0], remain[0][1] - anchor[1])
    cross = vec_diag[0] * vec_r0[1] - vec_diag[1] * vec_r0[0]

    if cross < 0:
        tr, bl = remain[0], remain[1]
    else:
        tr, bl = remain[1], remain[0]

    return np.array([anchor, tr, br, bl], dtype="float32")


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Anchor Mode: Tuning Sliders Enabled.")

    # Create Tuning Window
    cv.namedWindow("Tuning")
    cv.createTrackbar("Block Size", "Tuning", DEF_BLOCK_SIZE, 51, nothing)
    cv.createTrackbar("C Constant", "Tuning", DEF_C, 20, nothing)
    cv.createTrackbar("Min Area", "Tuning", DEF_MIN_AREA, 500, nothing)
    cv.createTrackbar("Circularity", "Tuning", DEF_CIRCULARITY, 100, nothing)

    # Force odd number for Block Size
    if DEF_BLOCK_SIZE % 2 == 0:
        cv.setTrackbarPos("Block Size", "Tuning", DEF_BLOCK_SIZE + 1)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Read Sliders
        bs = cv.getTrackbarPos("Block Size", "Tuning")
        c = cv.getTrackbarPos("C Constant", "Tuning")
        min_area = cv.getTrackbarPos("Min Area", "Tuning")
        circ_thresh = cv.getTrackbarPos("Circularity", "Tuning") / 100.0

        if bs < 3: bs = 3
        if bs % 2 == 0: bs += 1  # Block size must be odd

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 1. Adaptive Threshold (Controlled by Sliders)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, bs, c)

        # DISABLED MORPHOLOGY to prevent rings filling in.
        # If rings are broken, slightly increase Block Size instead.
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        # 2. Find Contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        anchors = []
        rings = []
        all_points = []

        if hierarchy is not None:
            hier = hierarchy[0]
            for i, cnt in enumerate(contours):
                area = cv.contourArea(cnt)

                # Area Filter (Slider controlled)
                if area < min_area or area > 5000: continue

                perimeter = cv.arcLength(cnt, True)
                if perimeter == 0: continue

                # Circularity Filter (Slider controlled)
                ratio = 4 * np.pi * (area / (perimeter * perimeter))
                if ratio < circ_thresh: continue

                M = cv.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Classification: Solid vs Hollow
                # hier[i][2] == -1 means NO CHILD (Solid)
                # hier[i][2] != -1 means HAS CHILD (Hollow Ring)

                point_type = "anchor" if hier[i][2] == -1 else "ring"

                if point_type == "anchor":
                    anchors.append((cx, cy))
                    # Draw Anchor in RED
                    cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv.putText(frame, "A", (cx + 5, cy), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    rings.append((cx, cy))
                    # Draw Ring in YELLOW
                    cv.circle(frame, (cx, cy), 4, (0, 255, 255), 1)
                    cv.putText(frame, "R", (cx + 5, cy), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                all_points.append((cx, cy))

        # --- MATCHING LOGIC (Same as before) ---
        for tl in anchors:
            if len(rings) < 3: break

            nearby_rings = []
            for r in rings:
                d = math.hypot(tl[0] - r[0], tl[1] - r[1])
                if d < CLUSTER_DIST:
                    nearby_rings.append((d, r))

            nearby_rings.sort(key=lambda x: x[0])
            candidates = [x[1] for x in nearby_rings[:6]]

            if len(candidates) < 3: continue

            card_found = False
            for ring_subset in itertools.combinations(candidates, 3):
                ordered_box = order_points_relative_to_anchor(tl, ring_subset)
                if ordered_box is not None:
                    tl_pt, tr, br, bl = ordered_box

                    w = np.linalg.norm(tr - tl_pt)
                    h = np.linalg.norm(bl - tl_pt)
                    if h == 0: continue
                    ar = w / h
                    if ar < 0.5 or ar > 2.5: continue

                    cos_tl = get_angle_cosine(tl_pt, tr, bl)
                    cos_tr = get_angle_cosine(tr, tl_pt, br)
                    cos_bl = get_angle_cosine(bl, tl_pt, br)
                    if max(cos_tl, cos_tr, cos_bl) > ANGLE_TOLERANCE: continue

                    box_int = np.int32(ordered_box)
                    has_internal_point = False
                    current_card_set = {tuple(tl), tuple(tr), tuple(br), tuple(bl)}

                    for pt in all_points:
                        if tuple(pt) not in current_card_set:
                            dist = cv.pointPolygonTest(box_int, (float(pt[0]), float(pt[1])), True)
                            if dist > 5.0:
                                has_internal_point = True
                                break

                    if has_internal_point: continue

                    cv.polylines(frame, [box_int], True, (0, 255, 0), 2)
                    cv.putText(frame, "CARD", (int(tl[0]), int(tl[1]) - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    card_found = True
                    break

            if card_found: continue

        cv.imshow('Anchor Detection', frame)
        # Display the Threshold view so you can see what the sliders are doing
        cv.imshow('Threshold View (Debug)', thresh)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()