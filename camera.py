import cv2

# Open the default camera (0), change to 1 or 2 if needed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Camera not accessible!")
else:
    print("✅ Camera detected! Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to read from camera.")
            break

        # Show the camera feed
        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
