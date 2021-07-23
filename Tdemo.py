# -*-coding:utf-8-*-
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import open3d as o3d
import time
import  xlwt
np.set_printoptions(threshold=np.inf)
IMG_WID = 640#图片的宽度
IMG_HGT = 480#图片的长度
#相机内参(width= 480, height= 640, fx= 460, fy= 460, cx=320, cy= 240)
#启动拍摄采集深度图像、彩色图像

# 配置深度和彩色数据流
start = time.time()#记录程序运行起始时间
pipeline = rs.pipeline()#定义通道变量，简化缩写
config = rs.config()#数据流配置简写，允许管道用户为管道流以及设备选择和配置请求过滤器
#config.enable_all_streams()#显示启用所有设备流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)#显示启用深度流，30是帧率
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)#显示启用彩色流
# #开始采集
pipeline.start(config)
# 深度图像向彩色对齐
align_to_color = rs.align(rs.stream.color)#简化缩写
def dianyuncjian(color_raw,depth_raw,A):
    # 默认转换功能从一对颜色和深度图像创建RGBDImage。彩色图像被转换为灰度图像，存储在[0，1]范围内。深度图像存储在中，以米为单位表示深度值。
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                    convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(width=480, height=640, fx=460, fy=460, cx=320, cy=240))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud("G:/ceshi/"+str(A)+".pcd", pcd)
    o3d.visualization.draw_geometries([pcd])
    return pcd

try:
    index = 0
    while True:
        # 等待连贯的帧:深度和颜色
        frames = pipeline.wait_for_frames()
        # 在上面稳定的帧上运行对齐算法以获得一组对齐的图像  对齐是指RGB+D
        frames = align_to_color.process(frames)
        # 从上面的对齐图像中抽出深度图像和彩色图像
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #判断是否两个图像是否有图像
        if not depth_frame or not color_frame:
            continue
        # 将RGB与D图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #将D图像转换为伪彩色图像
        # 在深度图像上应用伪彩色图像算法(图像必须通过cv2.convertScaleAbs转换为每像素8位)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # 将两幅图像举证就行行连接
        images = np.hstack((color_image, depth_colormap))
        # 合成的图像进行显示
        #cv2.namedWindow('冬青冠层图像采集软件', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Holly canopy image acquisition software', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('s'):
            index = index + 1
            cv2.imwrite("G:/ceshi/out/" + 'C' + str(index) + ".png", color_image)
            cv2.imwrite("G:/ceshi/out/" + 'D' + str(index) + ".png", depth_image)
            i=cv2.imread("G:/ceshi/out/" + 'D' + str(index) + ".png")
            CA = o3d.io.read_image("G:/ceshi/out/C" + str(index) + ".png")
            DA = o3d.io.read_image("G:/ceshi/out/D" + str(index) + ".png")

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(CA, DA,
                                                                            convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(width=480, height=640, fx=460, fy=460, cx=320, cy=240))
            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.io.write_point_cloud("G:/ceshi/out/" + str(index) + ".pcd", pcd)
            #o3d.visualization.draw_geometries([pcd])

            print(pcd)  # 输出点云点的个数
            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)  # aabb包围盒为红色

            aabb_box_length = np.asarray(aabb.get_extent())
            print("aabb包围盒的边长为：\n", aabb_box_length)

            #o3d.visualization.draw_geometries([pcd, aabb], window_name="植物株高分析")
            i1= cv2.imread("G:/ceshi/out/D" + str(index) + ".png",-1)
            i8 = i1 / (4000 - i1.min())
            i8 *= 255
            new = i8.astype(np.uint8)

            # print(np.mean(new))
            # print(new.max())
            r, rst = cv2.threshold(new, 180, 255, cv2.THRESH_BINARY_INV)
            #cv2.imshow("rst", rst)
            # 轮廓求找
            contours, hierarchy = cv2.findContours(rst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            m = len(contours)
            q = 0
            area = []
            for i in range(m):
                if len(contours[i]) > len(contours[q]):
                    q = i
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(contours[q])
            center = (int(x), int(y))
            radius = int(radius)
            # cv2.circle(new,center,radius,(0,255,0),3)
            y1 = int(y - radius)
            y2 = int(y + radius)
            x1 = int(x - radius)
            x2 = int(x + radius)
            jiequ = new[y1:y2, x1:x2]
            #cv2.imshow("jiequ", jiequ)

            img = jiequ.ravel()[np.flatnonzero(jiequ)]
            print(img.min())
            print(img.max())
            print(img.min())
            h_xiangsu= img.max()-img.min()
            k = 0.2643
            h = int(h_xiangsu/k)
            print("株高",h,"mm")


            '''def data_write(file_path, datas):
                f = xlwt.Workbook()
                sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
                # 将数据写入第 i 行，第 j 列
                i = 0
                for data in datas:
                    for j in range(len(data)):
                        sheet1.write(i, j, data[j])
                    i = i + 1
                f.save(file_path)  # 保存文件
            data_write("G:/ceshi/",h)'''

        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()



end = time.time()
print(end-start)