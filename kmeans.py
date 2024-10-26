import numpy as np
import cv2

def GetEuclideanDistance(Cbgr, Ibgr):
    return np.sqrt(np.sum((Cbgr - Ibgr) ** 2))

def kmeans_face_detection(img, k=2, max_iterations=100):
    centroids = np.array([img[np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])] for _ in range(k)], dtype=np.float32)

    for _ in range(max_iterations):
        labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        distances = np.zeros((img.shape[0], img.shape[1], k), dtype=np.float32)
        
        # Calcula a distância para cada centroid
        for c in range(k):
            distances[:, :, c] = np.linalg.norm(img - centroids[c], axis=2)

        labels = np.argmin(distances, axis=2)
        
        # Recalcula os centroids
        new_centroids = np.array([img[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c] for c in range(k)], dtype=np.float32)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Identifica o cluster da face com base em uma cor aproximada da pele
    skin_color_estimate = np.array([180, 130, 100])  # Aproximação do tom de pele
    face_cluster = np.argmin([np.linalg.norm(centroid - skin_color_estimate) for centroid in centroids])

    segmented_img = np.zeros_like(img)
    segmented_img[labels == face_cluster] = img[labels == face_cluster]  # Mantém apenas os pixels do cluster da face

    return segmented_img.astype(np.uint8)

if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE

    img = imread('res/unexistent-person-1', IMREAD_GRAYSCALE)

    img_kmeans = kmeans_face_detection(img=img, k=3, max_iterations=100) 

    imshow('basic', img_kmeans)
    waitKey(0)
    destroyAllWindows()
