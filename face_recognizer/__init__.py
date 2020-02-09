import torch
import numpy as np
from cache import lru
import face_recognition
from scipy.misc import imresize
from facial_embeddings.openface import get_facial_embeddings


class FaceRecognizer:

    def __init__(self, cache=None, threshold=0.5, embedding_dir="./models"):
        if cache:
            self.cache = cache
        else:
            self.cache = lru.LRUCache()

        self.threshold = threshold
        self.net, self.size = get_facial_embeddings(embedding_dir=embedding_dir)

    def recognize(self, img, identity):
        # data = torch.from_numpy(np.reshape(imresize(np.asarray(img, dtype="int32"), self.size), newshape=(1, 3, self.size[0], self.size[1]))).float()
        # embedding = self.net.forward(data)[0]
        data = np.asarray(img, dtype=np.uint8)
        embedding = face_recognition.face_encodings(data)[0]
        if self.cache.is_empty():
            self.cache.set(identity, embedding)
            return "Stranger"
        else:
            minimum_distance = float("inf")
            minimum_identity = "Stranger"

            for stored_identity in self.cache.dict:
                stored_embedding = self.cache.dict[stored_identity]
                distance = np.linalg.norm(stored_embedding - embedding)

                if minimum_distance > distance:
                    minimum_distance = distance
                    minimum_identity = stored_identity

            print(minimum_distance)

            if minimum_distance > self.threshold:
                self.cache.set(identity, embedding)
                return "Stranger"
            else:
                self.cache.set(minimum_identity, embedding)
                return minimum_identity
