import cv2
import glob


calibration_dir     = "pics_one_lane_x"
cal_imgs_paths = glob.glob(calibration_dir + "/*.png")
for path in range(0, len(cal_imgs_paths)):
    cal_img_path = cal_imgs_paths[path]
    img = cv2.imread(cal_img_path)
    resized_img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('pics_one_lane_x/frame1_'+str((path+1))+'.png', resized_img)