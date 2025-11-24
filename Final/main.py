import cv2 as cv
import numpy as np
import itertools
import math

# --- TUNING ---
MIN_RING_AREA = 100  # Min size of the corner ring
MAX_RING_AREA = 5000  # Max size
ASPECT_RATIO_TOL = 0.5  # 0.0 = Perfect Square. 0.5 allows rectangles.


def remove_close_points(points, min_dist=20):
    """
    Removes duplicate points that are too close to each other.
    """
    unique_points = []
    for p in points:
        is_close = False
        for up in unique_points:
            dist = math.hypot(p[0] - up[0], p[1] - up[1])
            if dist < min_dist:
                is_close = True
                break
        if not is_close:
            unique_points.append(p)
    return unique_points


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Looking for HOLLOW RINGS. Anti-Ghosting enabled.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Threshold: Adjust '11' and '2' if lighting is bad
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # 1. Find Contours with Hierarchy
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        raw_rings = []

        if hierarchy is not None:
            hier = hierarchy[0]
            for i, cnt in enumerate(contours):
                # Topology Filter: Must have a child (hole) -> hier[i][2] != -1
                if hier[i][2] == -1: continue

                # Size Filter
                area = cv.contourArea(cnt)
                if area < MIN_RING_AREA or area > MAX_RING_AREA: continue

                # Circularity Filter
                perimeter = cv.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.5: continue

                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    raw_rings.append((cx, cy))
                    # Debug: Show rings in yellow
                    cv.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

        # 2. Filter duplicate rings
        corner_rings = remove_close_points(raw_rings)

        # 3. Find Candidates
        candidates = []

        if len(corner_rings) >= 4:
            indexed_rings = list(enumerate(corner_rings))
            search_list = indexed_rings[:16]

            for quad in itertools.combinations(search_list, 4):
                indices = {item[0] for item in quad}
                pts = np.array([item[1] for item in quad], dtype="float32")

                # Sort points: TL, TR, BR, BL
                s = pts.sum(axis=1)
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]

                # Metric 1: Geometry
                width_top = np.linalg.norm(tr - tl)
                width_bot = np.linalg.norm(br - bl)
                height_left = np.linalg.norm(bl - tl)
                height_right = np.linalg.norm(br - tr)

                diff_w = abs(width_top - width_bot)
                diff_h = abs(height_left - height_right)

                if diff_w > 50 or diff_h > 50: continue

                # Metric 2: Area
                area = width_top * height_left
                if area < 5000 or area > 100000: continue

                # Metric 3: INTERNAL CHECK (Anti-Ghosting)
                # Ensure no other detected rings exist INSIDE this rectangle
                box_int = np.array([tl, tr, br, bl], np.int32)
                has_internal_points = False

                for idx, r_pt in indexed_rings:
                    if idx not in indices:  # If this ring is not one of the corners of the box
                        # Check if it lies inside the box
                        # pointPolygonTest returns > 0 if inside
                        if cv.pointPolygonTest(box_int, (float(r_pt[0]), float(r_pt[1])), False) > 0:
                            has_internal_points = True
                            break

                if has_internal_points:
                    continue  # Reject this big rectangle, it swallowed another card's corner!

                score = diff_w + diff_h

                candidates.append({
                    'score': score,
                    'box': box_int,
                    'indices': indices,
                    'center': ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
                })

        # 4. SOLVE OVERLAPS
        candidates.sort(key=lambda x: x['score'])
        used_indices = set()

        for cand in candidates:
            if not used_indices.intersection(cand['indices']):
                cv.polylines(frame, [cand['box']], True, (0, 255, 0), 3)
                cv.putText(frame, "VALID CARD", (int(cand['center'][0] - 60), int(cand['center'][1])),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                used_indices.update(cand['indices'])

        cv.imshow('Non-Overlapping Detection', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()