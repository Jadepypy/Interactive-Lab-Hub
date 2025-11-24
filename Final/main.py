import cv2 as cv
import numpy as np
import math

# --- CONFIGURATION ---
DOT_SIZE_MIN = 20
DOT_SIZE_MAX = 2000
CLUSTER_DIST = 60  # Max pixels between dots to be part of one corner
ALIGNMENT_STRICTNESS = 0.9  # 1.0 = Perfect line, 0.9 = Allow ~25 deg deviation
CARD_MAX_DIM = 800  # Max width/height of card in pixels


def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_vec(start, end):
    """ Returns a normalized unit vector (x, y) from start to end. """
    vx, vy = end[0] - start[0], end[1] - start[1]
    mag = math.hypot(vx, vy)
    if mag == 0: return (0, 0)
    return (vx / mag, vy / mag)


def get_angle(v1, v2):
    """ Returns angle in degrees between two vectors. """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    # Clip for float safety
    cos_angle = np.clip(dot, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def classify_corner_shape(cluster):
    """
    Analyzes 3 dots.
    Returns:
        CornerType ("TL", "TR", "BL", "BR"),
        Vertex (x,y),
        Arms [vec_horizontal_ish, vec_vertical_ish]
    """
    p0, p1, p2 = cluster

    # 1. Identify Vertex (Opposite longest side)
    d01 = get_dist(p0, p1)
    d12 = get_dist(p1, p2)
    d20 = get_dist(p2, p0)

    sides = [(d12, p0, p1, p2), (d20, p1, p0, p2), (d01, p2, p0, p1)]
    sides.sort(key=lambda x: x[0])  # Longest first? No, longest is hypotenuse.
    # We want hypotenuse last to find vertex.
    # sides[0] = shortest side
    # sides[1] = middle side
    # sides[2] = longest side (hypotenuse)

    hypotenuse, vertex, arm_end1, arm_end2 = sides[2]

    # 2. Calculate Arm Vectors
    v1 = get_vec(vertex, arm_end1)
    v2 = get_vec(vertex, arm_end2)

    # 3. Check Angle (Must be roughly 90 deg)
    angle = get_angle(v1, v2)
    if angle < 70 or angle > 110:
        return None, None, None

    # 4. Determine Orientation (TL, TR, etc)
    # Vector Sum points towards the center of the card
    diag_x = v1[0] + v2[0]
    diag_y = v1[1] + v2[1]

    c_type = None
    # We assume camera is roughly upright (+/- 45 deg tilt)
    if diag_x > 0 and diag_y > 0:
        c_type = "TL"
    elif diag_x < 0 and diag_y > 0:
        c_type = "TR"
    elif diag_x < 0 and diag_y < 0:
        c_type = "BR"
    elif diag_x > 0 and diag_y < 0:
        c_type = "BL"

    if not c_type: return None, None, None

    # 5. Sort vectors into [Right-ish/Left-ish, Down-ish/Up-ish]
    # This helps matching later.
    # For TL: Arms are Right and Down.
    # For TR: Arms are Left and Down.
    arms = [v1, v2]
    arms.sort(key=lambda v: abs(v[0]), reverse=True)  # Sort by X-component magnitude
    # arms[0] is the more horizontal one, arms[1] is the more vertical one

    return c_type, vertex, arms


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Robust Pattern Matching Started.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # --- 1. FIND RAW DOTS ---
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
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    all_dots.append((cx, cy))

        # --- 2. IDENTIFY CORNERS (L-SHAPES) ---
        corners = {"TL": [], "TR": [], "BL": [], "BR": []}
        used_dots = [False] * len(all_dots)

        for i in range(len(all_dots)):
            if used_dots[i]: continue

            # Simple neighbor search
            cluster = [all_dots[i]]
            for j in range(i + 1, len(all_dots)):
                if not used_dots[j] and get_dist(all_dots[i], all_dots[j]) < CLUSTER_DIST:
                    cluster.append(all_dots[j])

            if len(cluster) == 3:
                c_type, vertex, arms = classify_corner_shape(cluster)
                if c_type:
                    corners[c_type].append({'pt': vertex, 'arms': arms})

                    # VISUALIZATION OF CORNERS
                    color = (0, 255, 0)  # Default Green
                    if c_type == "TR":
                        color = (255, 0, 0)  # Blue
                    elif c_type == "BL":
                        color = (0, 255, 255)  # Yellow
                    elif c_type == "BR":
                        color = (0, 0, 255)  # Red

                    cv.circle(frame, vertex, 5, color, -1)
                    # Draw Arms: White lines showing detection direction
                    end1 = (int(vertex[0] + arms[0][0] * 20), int(vertex[1] + arms[0][1] * 20))
                    end2 = (int(vertex[0] + arms[1][0] * 20), int(vertex[1] + arms[1][1] * 20))
                    cv.line(frame, vertex, end1, (255, 255, 255), 1)
                    cv.line(frame, vertex, end2, (255, 255, 255), 1)
                    cv.putText(frame, c_type, (vertex[0] - 10, vertex[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # --- 3. MATCH CORNERS (ROTATION ROBUST) ---
        # We start with TL and look for neighbors along its arms

        used_tr, used_bl, used_br = [], [], []

        for tl_data in corners["TL"]:
            tl = tl_data['pt']
            tl_h_vec = tl_data['arms'][0]  # Expected Horizontal direction
            tl_v_vec = tl_data['arms'][1]  # Expected Vertical direction

            best_tr = None
            best_bl = None

            # A. Find TR
            # It must be in the direction of the TL's horizontal arm
            for tr_data in corners["TR"]:
                tr = tr_data['pt']
                vec_to_tr = get_vec(tl, tr)
                dist = get_dist(tl, tr)
                if dist > CARD_MAX_DIM: continue

                # Check Alignment: Dot product should be close to 1
                alignment = (vec_to_tr[0] * tl_h_vec[0]) + (vec_to_tr[1] * tl_h_vec[1])
                if alignment > ALIGNMENT_STRICTNESS:
                    best_tr = tr
                    break  # Found a matching TR

            # B. Find BL
            # It must be in the direction of the TL's vertical arm
            for bl_data in corners["BL"]:
                bl = bl_data['pt']
                vec_to_bl = get_vec(tl, bl)
                dist = get_dist(tl, bl)
                if dist > CARD_MAX_DIM: continue

                alignment = (vec_to_bl[0] * tl_v_vec[0]) + (vec_to_bl[1] * tl_v_vec[1])
                if alignment > ALIGNMENT_STRICTNESS:
                    best_bl = bl
                    break

            # C. If we have TL, TR, and BL, check for BR
            if best_tr and best_bl:
                # Predict where BR should be (Parallelogram logic)
                # BR = TR + (BL - TL)
                pred_br_x = best_tr[0] + (best_bl[0] - tl[0])
                pred_br_y = best_tr[1] + (best_bl[1] - tl[1])

                # Look for a real BR corner near this prediction
                for br_data in corners["BR"]:
                    br = br_data['pt']
                    dist_err = math.hypot(br[0] - pred_br_x, br[1] - pred_br_y)

                    if dist_err < 50:  # Tolerance
                        # SUCCESS: Found all 4 corners
                        pts = np.array([tl, best_tr, br, best_bl], np.int32)
                        cv.polylines(frame, [pts], True, (0, 255, 0), 2)

                        # Calculate Center
                        cx = int((tl[0] + br[0]) / 2)
                        cy = int((tl[1] + br[1]) / 2)
                        cv.putText(frame, "CARD", (cx - 20, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break

        cv.imshow('Robust Reader', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()