import cv2 as cv
import numpy as np
import itertools
import math

# --- TUNING ---
MIN_RING_AREA = 100  # Min size of the corner ring
MAX_RING_AREA = 5000  # Max size
MIN_EDGE_LENGTH = 100  # Min pixels for any side of the card (prevents detecting thin gaps)
ANGLE_TOLERANCE = 0.25  # Cosine val. 0.0 is 90deg. 0.25 allows ~75-105deg.


def remove_close_points(points, min_dist=20):
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


def get_angle_cosine(vertex, p1, p2):
    """
    Calculates the cosine of the angle at 'vertex' formed by p1-vertex-p2.
    Returns value close to 0 if angle is 90 degrees.
    """
    # Vector v1: vertex -> p1
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    # Vector v2: vertex -> p2
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 == 0 or mag2 == 0: return 1.0  # Invalid
    return abs(dot / (mag1 * mag2))


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Looking for HOLLOW RINGS. Strict Geometry (Angles + Edges) enabled.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        raw_rings = []

        if hierarchy is not None:
            hier = hierarchy[0]
            for i, cnt in enumerate(contours):
                if hier[i][2] == -1: continue  # Must have child (hole)

                area = cv.contourArea(cnt)
                if area < MIN_RING_AREA or area > MAX_RING_AREA: continue

                perimeter = cv.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.5: continue

                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    raw_rings.append((cx, cy))
                    cv.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

        corner_rings = remove_close_points(raw_rings)
        candidates = []

        if len(corner_rings) >= 4:
            indexed_rings = list(enumerate(corner_rings))
            # Optimization: Sort rings by X coordinate to only check neighbors?
            # For now, limiting list size is safer for stability.
            search_list = indexed_rings[:16]

            for quad in itertools.combinations(search_list, 4):
                indices = {item[0] for item in quad}
                pts = np.array([item[1] for item in quad], dtype="float32")

                # 1. Sort points (TL, TR, BR, BL)
                s = pts.sum(axis=1)
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]

                # 2. Side Length Check (Prevents thin gaps or clusters)
                w1 = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
                w2 = math.hypot(br[0] - bl[0], br[1] - bl[1])
                h1 = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
                h2 = math.hypot(br[0] - tr[0], br[1] - tr[1])

                if min(w1, w2, h1, h2) < MIN_EDGE_LENGTH: continue

                # 3. Angle Check (Strict 90-degree Check)
                # Calculates cosine of angle at each corner. Must be close to 0.
                cos_tl = get_angle_cosine(tl, tr, bl)
                cos_tr = get_angle_cosine(tr, tl, br)
                cos_br = get_angle_cosine(br, tr, bl)
                cos_bl = get_angle_cosine(bl, tl, br)

                if (cos_tl > ANGLE_TOLERANCE or cos_tr > ANGLE_TOLERANCE or
                        cos_br > ANGLE_TOLERANCE or cos_bl > ANGLE_TOLERANCE):
                    continue

                # 4. Aspect Ratio Check (Prevent extremely long/weird shapes)
                avg_w = (w1 + w2) / 2
                avg_h = (h1 + h2) / 2
                aspect = avg_w / avg_h if avg_h > 0 else 0
                # Assuming cards are somewhat standard (e.g. 3:2 ratio = 1.5)
                # Allow range 0.5 (tall) to 2.5 (wide)
                if aspect < 0.4 or aspect > 2.5: continue

                # 5. Internal Point Check (Anti-Ghosting)
                box_int = np.array([tl, tr, br, bl], np.int32)
                has_internal = False
                for idx, r_pt in indexed_rings:
                    if idx not in indices:
                        if cv.pointPolygonTest(box_int, (float(r_pt[0]), float(r_pt[1])), False) > 0:
                            has_internal = True
                            break
                if has_internal: continue

                # Calculate Score (Closer to perfect rectangle = lower score)
                # Score = (diff in widths) + (diff in heights) + (sum of angle deviations)
                angle_score = (cos_tl + cos_tr + cos_br + cos_bl) * 100
                geom_score = abs(w1 - w2) + abs(h1 - h2)
                total_score = geom_score + angle_score

                candidates.append({
                    'score': total_score,
                    'box': box_int,
                    'indices': indices,
                    'center': ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
                })

        # Solve Overlaps
        candidates.sort(key=lambda x: x['score'])
        used_indices = set()

        for cand in candidates:
            if not used_indices.intersection(cand['indices']):
                cv.polylines(frame, [cand['box']], True, (0, 255, 0), 3)
                cv.putText(frame, "ID Ready", (int(cand['center'][0] - 40), int(cand['center'][1])),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                used_indices.update(cand['indices'])

        cv.imshow('Strict Geometry Detection', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()