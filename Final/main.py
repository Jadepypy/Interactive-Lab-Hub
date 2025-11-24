import cv2 as cv
import numpy as np
import math

# --- CONSTANTS (Tune these for your camera/height) ---
# Maximum distance between dots to be considered part of the SAME CORNER
CORNER_CLUSTER_DIST = 60
# Maximum distance between corners to be considered part of the SAME CARD
CARD_CLUSTER_DIST = 400


def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def cluster_points(points, max_dist):
    """
    Simple clustering: groups points that are within max_dist of each other.
    Returns a list of clusters (each cluster is a list of points).
    """
    clusters = []
    # Track which points have been processed
    processed = [False] * len(points)

    for i in range(len(points)):
        if processed[i]:
            continue

        # Start a new cluster with this point
        current_cluster = [points[i]]
        processed[i] = True

        # Keep adding neighbors to this cluster
        # (Simple greedy approach is sufficient for this number of points)
        stack = [points[i]]
        while stack:
            ref_pt = stack.pop()
            for j in range(len(points)):
                if not processed[j]:
                    dist = get_dist(ref_pt, points[j])
                    if dist < max_dist:
                        current_cluster.append(points[j])
                        processed[j] = True
                        stack.append(points[j])

        clusters.append(current_cluster)
    return clusters


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Searching for cards with STRICTLY 3 dots per corner...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 1. Threshold
        # Using adaptive threshold can be more robust for outlines/rings
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)

        # 2. Find All Dots
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        all_dots = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 20 or area > 2000: continue  # Size filter

            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity > 0.6:  # Relaxed slightly for rings
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    all_dots.append((cx, cy))
                    # Draw small red dot for raw detection
                    cv.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        # 3. STAGE 1: Group Dots into "Corners"
        # We expect a corner to be a tight cluster of 3 dots
        potential_corners = cluster_points(all_dots, CORNER_CLUSTER_DIST)
        valid_corners = []

        for cluster in potential_corners:
            # STRICT REQUIREMENT: Corner must have exactly 3 dots
            if len(cluster) == 3:
                # Calculate center of this corner (average of 3 dots)
                avg_x = int(sum(p[0] for p in cluster) / 3)
                avg_y = int(sum(p[1] for p in cluster) / 3)
                valid_corners.append((avg_x, avg_y))

                # Visual: Draw a blue circle around the valid corner group
                cv.circle(frame, (avg_x, avg_y), 25, (255, 255, 0), 2)
            else:
                # Optional: Visualize invalid clusters (e.g. noise) in gray
                for p in cluster:
                    cv.circle(frame, p, 5, (100, 100, 100), 1)

        # 4. STAGE 2: Group "Corners" into "Cards"
        # We expect a card to be a group of 4 corners
        potential_cards = cluster_points(valid_corners, CARD_CLUSTER_DIST)

        for card_corners in potential_cards:
            # STRICT REQUIREMENT: Card must have exactly 4 corners
            if len(card_corners) == 4:
                # We found a valid card!
                # (4 corners * 3 dots/corner = 12 dots total confirmed)

                # Calculate boundary box for these 4 corners
                points_array = np.array(card_corners)
                rect = cv.minAreaRect(points_array)
                box = cv.boxPoints(rect)
                box = np.int32(box)

                # Visual: Green Boundary for confirmed cards
                cv.drawContours(frame, [box], 0, (0, 255, 0), 3)

                # Label
                cx, cy = int(rect[0][0]), int(rect[0][1])
                cv.putText(frame, "VALID CARD", (cx - 50, cy),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow('Strict Card Detection', frame)
        cv.imshow('Threshold', thresh)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()