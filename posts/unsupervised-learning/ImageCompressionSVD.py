import numpy as np
import matplotlib.pyplot as plt

class ImageCompressionSVD:

    @staticmethod
    def to_grayscale_image(img):
        return 1 - np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def compress_image(img, k, epsilon = None):
        # u, s, v = np.linalg.svd(img)
        # return u[:, :k] @ np.diag(s[:k]) @ v[:k, :]
        u, s, v = np.linalg.svd(img)
        if epsilon is not None:
            s[s < epsilon] = 0
        return u[:, :k] @ np.diag(s[:k]) @ v[:k, :]
    
    @staticmethod
    def compare_images(A, A_, title_1 = "original image", title_2 = "reconstructed image"):
        fig, axarr = plt.subplots(1, 2, figsize = (7, 3))

        axarr[0].imshow(A, cmap = "Greys")
        axarr[0].axis("off")
        axarr[0].set(title = title_1)

        axarr[1].imshow(A_, cmap = "Greys")
        axarr[1].axis("off")
        axarr[1].set(title = title_2)

    @staticmethod
    def get_relative_size(A, k):
        A_space = A.shape[0] * A.shape[1]
        A_compressed_space = k * (A.shape[0] + A.shape[1]) + k
        return A_compressed_space / A_space
    
    @staticmethod
    def find_k_for_compression_threshold(img, compression_factor):
        if compression_factor <= 1: raise ValueError("compression factor must be greater than 1")

        k = 1
        threshold = 1 / compression_factor
        print(f"threshold: {threshold}")
        MAX_ITER = min(img.shape[0], img.shape[1])

        # Find the max k such that the relative size is less than the threshold
        while k < MAX_ITER:
            if ImageCompressionSVD.get_relative_size(img, k) > threshold:
                k = None
                break
            if ImageCompressionSVD.get_relative_size(img, k+1) > threshold:
                break
            k += 1

        return k
        
    @classmethod
    def svd_reconstruct(cls, img, k, epsilon = None):
        A = cls.to_grayscale_image(img)
        A_ = cls.compress_image(A, k, epsilon)
        cls.compare_images(A, A_)
        plt.show()

    @classmethod
    def svd_experiment(cls, img, k_values):
        A = cls.to_grayscale_image(img)
        for k in k_values:
            A_ = cls.compress_image(A, k)
            print(f"relative size for k = {k}: {cls.get_relative_size(A, k)}")
            cls.compare_images(A, A_)
            plt.show()

    