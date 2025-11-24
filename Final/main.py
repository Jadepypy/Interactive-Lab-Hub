import cv2 as cv
import numpy as np
import math

# --- TUNING ---
MIN_AREA = 50
MAX_AREA = 2500
CLUSTER_DIST = 200
ANGLE_TOLERANCE = 0.25


def get_angle_cosine(vertex, p1, p2):
    """ Calculates how close the angle at 'vertex' is to 90 degrees. """
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0: return 1.0
    return abs(dot / (mag1 * mag2))


def order_points_relative_to_anchor(anchor, neighbors):
    """ Identifies TR, BR, BL relative to the Top-Left Anchor. """
    pts = [anchor] + neighbors

    # 1. Find BR (Farthest from Anchor)
    dists = [math.hypot(p[0] - anchor[0], p[1] - anchor[1]) for p in pts]
    br_idx = np.argmax(dists)
    br = pts[br_idx]

    remain = [p for i, p in enumerate(pts) if i != 0 and i != br_idx]
    if len(remain) != 2: return None

    # 2. Distinguish TR (Right) vs BL (Down) using Cross Product
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

    print("Anchor Mode: Robust (Morphology Enabled).")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 1. Adaptive Threshold (Handles uneven lighting)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # 2. MORPHOLOGICAL CLOSING (The Fix for Broken Rings)
        # This connects gaps in the ink so "C" shapes become "O" shapes
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        # 3. Find Contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        anchors = []  # Solid Dots
        rings = []  # Hollow Rings
        all_points = []

        if hierarchy is not None:
            hier = hierarchy[0]
            for i, cnt in enumerate(contours):
                area = cv.contourArea(cnt)
                if area < MIN_AREA or area > MAX_AREA: continue

                perimeter = cv.arcLength(cnt, True)
                if perimeter == 0: continue

                # Check circularity to avoid noise
                ratio = area / (perimeter * perimeter)
                if ratio < 0.04: continue

                M = cv.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # IMPORTANT: hier[i][2] != -1 means it HAS a child (A hole inside)
                point_type = "anchor" if hier[i][2] == -1 else "ring"

                if point_type == "anchor":
                    anchors.append((cx, cy))
                    cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Red Dot
                else:
                    rings.append((cx, cy))
                    cv.circle(frame, (cx, cy), 4, (0, 255, 255), 1)  # Yellow Ring (Empty center)

                all_points.append((cx, cy))

        # --- MATCHING ---
        for tl in anchors:
            if len(rings) < 3: break

            # Find 3 closest rings
            ring_dists = []
            for r in rings:
                d = math.hypot(tl[0] - r[0], tl[1] - r[1])
                if d < CLUSTER_DIST:
                    ring_dists.append((d, r))

            # Optimization: Only sort if we found enough neighbors
            if len(ring_dists) < 3: continue

            ring_dists.sort(key=lambda x: x[0])
            closest_3 = [x[1] for x in ring_dists[:3]]

            ordered_box = order_points_relative_to_anchor(tl, closest_3)

            if ordered_box is not None:
                tl_pt, tr, br, bl = ordered_box

                # --- GEOMETRY CHECKS ---
                w = np.linalg.norm(tr - tl_pt)
                h = np.linalg.norm(bl - tl_pt)
                if h == 0: continue

                # Aspect Ratio
                ar = w / h
                if ar < 0.5 or ar > 2.5: continue

                # Angle Check
                cos_tl = get_angle_cosine(tl_pt, tr, bl)
                cos_tr = get_angle_cosine(tr, tl_pt, br)
                cos_bl = get_angle_cosine(bl, tl_pt, br)
                if max(cos_tl, cos_tr, cos_bl) > ANGLE_TOLERANCE:
                    continue

                # --- EMPTY INTERIOR CHECK ---
                box_int = np.int32(ordered_box)
                has_internal_point = False
                current_card_set = {tuple(tl), tuple(tr), tuple(br), tuple(bl)}

                for pt in all_points:
                    if tuple(pt) not in current_card_set:
                        if cv.pointPolygonTest(box_int, (float(pt[0]), float(pt[1])), False) > 0:
                            has_internal_point = True
                            break

                if has_internal_point: continue

                # --- SUCCESS ---
                cv.polylines(frame, [box_int], True, (0, 255, 0), 2)

                # Show ID dot locations (Debugging)
                # You can add logic here to extract the ID
                cv.putText(frame, "CARD", (int(tl[0]), int(tl[1]) - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow('Anchor Detection', frame)
        # cv.imshow('Thresh', thresh) # Uncomment to debug ink gaps

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()