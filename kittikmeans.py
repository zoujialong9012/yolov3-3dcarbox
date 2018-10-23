import numpy as np
from PIL import Image


from kitti_data_processing import get_random_data, kitti_parse_annotation 

class YOLO_Kmeans:

    def __init__(self, cluster_number, label_dir, image_dir):
        self.cluster_number = cluster_number
        self.label_dir = label_dir
        self.image_dir = image_dir

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("kitti_yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()


    def txt2clusters(self):
        #all_boxes = self.txt2boxes()
        w = 960
        h = 384
        dataSet = []
        all_images = kitti_parse_annotation(self.label_dir, self.image_dir)
        for image_objs in all_images:
            obj_cnt = 0
            for obj in image_objs:
               if obj_cnt == 0:
                  image = Image.open(obj['image'])
                  iw, ih = image.size
                  scale = min(w/iw, h/ih)
                  image.close()
               
               nw = (obj['xmax'] - obj['xmin'])*scale
               nh = (obj['ymax'] - obj['ymin'])*scale
               dataSet.append([nw, nh])

        all_boxes = np.array(dataSet) 
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    label_dir = '/home/cidi/dl/3dobject/3D-Deepbox/training/label_2/'
    image_dir = '/home/cidi/dl/3dobject/3D-Deepbox/training/image_2/'
    kmeans = YOLO_Kmeans(cluster_number, label_dir, image_dir)
    kmeans.txt2clusters()
