import cv2 as cv
import numpy as np


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Convert to Grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 2. Threshold to find BLACK dots
        # Binary Inverse: Black (dark) becomes White (255), everything else Black (0)
        # You might need to adjust '100' depending on your room's brightness
        _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

        # 3. Find Contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        detected_dots = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            perimeter = cv.arcLength(cnt, True)

            # Filter 1: Size Check
            # We only want "little circles". Ignore tiny noise (<50) or big objects (>1000)
            if area < 50 or area > 1000:
                continue

            # Filter 2: Circularity Check
            # A perfect circle has a circularity of ~1.0. A square is ~0.78.
            # We calculate this to ensure we don't pick up random scuff marks.
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # If it is circular enough (e.g. > 0.7), we assume it's one of our dots
            if circularity > 0.7:
                # Find the center of the dot
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Add to our list of points
                    detected_dots.append([cx, cy])

                    # Draw a green dot on the screen so you know it was detected
                    cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # 4. Create the Boundary
        # If we found any dots, draw a rectangle that encompasses ALL of them
        if detected_dots:
            points_array = np.array(detected_dots)
            x, y, w, h = cv.boundingRect(points_array)

            # Add a little padding so the box isn't touching the dots exactly
            pad = 20
            cv.rectangle(frame, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 2)
            cv.putText(frame, "Paper Boundary", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the result
        cv.imshow('Dot Detection', frame)

        # Optional: Show the black/white view to help debug lighting
        # (White blobs here are what the computer thinks are "dots")
        cv.imshow('Threshold View', thresh)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()