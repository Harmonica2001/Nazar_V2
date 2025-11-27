import cv2
import numpy as np
import math

# ----- Parameters -----
image_path = r"C:\Users\ahmad\Nazar\Repo\Nazar_V2\Segmentation_Gate_photos_BB\20251127_152712_831289_cls79_flag1_bb.jpg"
# Hough params
canny_low = 50
canny_high = 150
blur_k = 5  # must be odd
rho = 1
theta = np.pi/180
hough_threshold = 50
min_line_len = 40
max_line_gap = 10
# Parallel tolerance
ANGLE_TOL_DEG = 5.0
# -----------------------

def line_angle_rad(x1, y1, x2, y2):
    return math.atan2((y2 - y1), (x2 - x1))  # [-pi, pi]

def angle_deg_between(a, b):
    diff = abs(a - b) % math.pi
    if diff > math.pi/2:
        diff = math.pi - diff
    return math.degrees(diff)

def line_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def point_line_distance(px, py, x1, y1, x2, y2):
    # perpendicular distance from point (px,py) to line segment (x1,y1)-(x2,y2) infinite line form
    # If segment needed, we can clamp; here we use infinite line distance
    num = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den != 0 else float('inf')

def avg_distance_between_lines(l1, l2, sample_points=5):
    # l: (x1,y1,x2,y2)
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2

    def sample_points_on_segment(x1, y1, x2, y2, n):
        pts = []
        if n <= 1:
            pts.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            return pts
        for i in range(n):
            t = i / (n - 1)
            px = x1 + (x2 - x1) * t
            py = y1 + (y2 - y1) * t
            pts.append((px, py))
        return pts

    pts1 = sample_points_on_segment(x11, y11, x12, y12, sample_points)
    pts2 = sample_points_on_segment(x21, y21, x22, y22, sample_points)

    d1 = [point_line_distance(px, py, x21, y21, x22, y22) for px, py in pts1]
    d2 = [point_line_distance(px, py, x11, y11, x12, y12) for px, py in pts2]

    avg1 = sum(d1) / len(d1) if d1 else 0.0
    avg2 = sum(d2) / len(d2) if d2 else 0.0
    return (avg1 + avg2) / 2.0

def find_parallel_pairs(lines, angle_tol_deg=ANGLE_TOL_DEG):
    pairs = []
    n = len(lines)
    angles = [line_angle_rad(*l[0]) for l in lines]  # lines from Hough are [[[x1,y1,x2,y2]], ...]
    for i in range(n):
        x1,y1,x2,y2 = lines[i][0]
        for j in range(i+1, n):
            x3,y3,x4,y4 = lines[j][0]
            a1 = angles[i]
            a2 = angles[j]
            ang_diff = angle_deg_between(a1, a2)
            if ang_diff <= angle_tol_deg:
                pairs.append(((x1,y1,x2,y2),(x3,y3,x4,y4), ang_diff))
    return pairs

def main():
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    lines = cv2.HoughLinesP(edges, rho, theta, hough_threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        print("No lines found.")
        return

    pairs = find_parallel_pairs(lines, ANGLE_TOL_DEG)
    print(f"Found {len(lines)} lines, {len(pairs)} parallel pairs (<= {ANGLE_TOL_DEG}°)")

    results = []
    for idx, (l1, l2, ang_diff) in enumerate(pairs, 1):
        x1,y1,x2,y2 = l1
        x3,y3,x4,y4 = l2
        len1 = line_length(x1,y1,x2,y2)
        len2 = line_length(x3,y3,x4,y4)
        ratio = len1 / len2 if len2 != 0 else float('inf')
        avg_dist = avg_distance_between_lines(l1, l2, sample_points=9)
        parallelism_score = max(0.0, 1.0 - (ang_diff / ANGLE_TOL_DEG))  # 1.0 if ang_diff=0, 0 if ang_diff==ANGLE_TOL_DEG
        results.append({
            "pair_index": idx,
            "angle_diff_deg": ang_diff,
            "len1": len1,
            "len2": len2,
            "length_ratio": ratio,
            "avg_distance": avg_dist,
            "parallelism_score": parallelism_score,
            "l1": l1,
            "l2": l2
        })
        print(f"Pair {idx}: angle_diff={ang_diff:.2f}°, len1={len1:.1f}, len2={len2:.1f}, ratio={ratio:.3f}, avg_dist={avg_dist:.2f}, score={parallelism_score:.3f}")

    # Visualization: draw lines and pair indices
    vis = orig.copy()
    for r in results:
        x1,y1,x2,y2 = r["l1"]
        x3,y3,x4,y4 = r["l2"]
        cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.line(vis, (x3,y3), (x4,y4), (255,0,0), 2)
        # draw midpoints and label with pair index and score
        mx1, my1 = int((x1+x2)/2), int((y1+y2)/2)
        mx2, my2 = int((x3+x4)/2), int((y3+y4)/2)
        cv2.circle(vis, (mx1,my1), 3, (0,255,0), -1)
        cv2.circle(vis, (mx2,my2), 3, (255,0,0), -1)
        label = f"#{r['pair_index']} s={r['parallelism_score']:.2f}"
        cv2.putText(vis, label, (mx1+5, my1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    # show
    cv2.namedWindow("Lines & Pairs", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Lines & Pairs", vis)
    cv2.namedWindow("Edges", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Edges", edges)
    print("\nDone. Close window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




