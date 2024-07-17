# Algorithm
import numpy as np
import cv2
import scipy.linalg as s_linalg


class pca_class:
   
    def __init__(self, images, y, folder_names, no_of_elements, quality_percent):
        '''
        images: image matrix
        y: image labels
        folder: image target folder
        quality percent: how much quality to retain
        '''
        self.no_of_elements = no_of_elements #no of people, folders
        self.images = np.asarray(images)
        self.y = y
        self.folder_names = folder_names 

        # Centers each image in self.images around the mean face,
        mean = np.mean(self.images, 1) # 1 because we are finding mean along columns. 0 would have meant rows. We take mean of every image in matrix
        self.mean_face = np.asmatrix(mean).T
        self.images = self.images - self.mean_face #centering the original image matrix around mean
        self.quality_percent = quality_percent

    def reduce_dim(self):

        U, sigma, VT = s_linalg.svd(self.images, full_matrices=True)
        '''
        U and VT are orthogonal.
        U has the normalized eigen vectors. Left singular vectors
        VT is the transpose of right singular vectors of the image matrix
        sigma would have the singular values of images matrix.
        
        inshort, p q and d are U, sigma and V^T in the SVD formula.
        '''
        u_matrix = np.matrix(U)
        sigma_diag = np.diag(sigma)
        VT_matrix = np.matrix(VT)

        p = self.give_p(sigma) 
        self.new_bases = u_matrix[:, 0:p] #select only important vectors(cols)
        self.new_coordinates = np.dot(self.new_bases.T, self.images) 
        return self.new_coordinates.T
    
    def give_p(self, d):
        '''
        Tells you how many singular values are needed to capture the quality 
        '''
        sum = np.sum(d) #sum of all eigen values.
        sum_85 = self.quality_percent * sum/100
        temp = 0
        p = 0
        while temp < sum_85:
            temp += d[p]
            p += 1
        return p

    def new_cord_for_image(self, image):
        '''
        Project the new image (converted to an array) onto the same principal components we got earlier
        new_mean = ((old mean of face x number of old images) + image vector ) / len(labels) + 1
       
        Lets say my original matrix was 4x4.
        The first row was  1st row was [1 2 3 4]
        The mean value corresponding to this in the vector would be 2.5 or 10/4

        A 4x4 matrix means there are 4 images of 4 pixels each.

        So multiplying 2.5 with 4 (number of labels) scales it back to the total sum of the pixel values of all images that is 10?
        
        Now we add the new image vector to this and divide again by 4+1 because of one new image. 
         '''
        img_vec = np.asmatrix(image).ravel() #convert to vector
        img_vec = img_vec.T # make it column vector
        new_mean = ((self.mean_face * len(self.y)) + img_vec) / (len(self.y) + 1)
        img_vec = img_vec - new_mean #subtract mean from image as always
        return np.dot(self.new_bases.T, img_vec)

    def recognize_face(self, new_cord_pca, k=0):
        classes = len(self.no_of_elements)
        start = 0
        distances = []
        for i in range(classes):
            # Retrieve the PCA coordinates(reduced dim) of all images belonging to the current class(folder) i.
            temp_imgs = self.new_coordinates[:, int(start): int(start + self.no_of_elements[i])] 
            # Compute the mean PCA coordinates of the current class i along columns.
            mean_temp = np.mean(temp_imgs, 1)
            start = start + self.no_of_elements[i]

            # Calculates the Euclidean distance between the PCA coordinates of the new test face (new_cord_pca) and the mean PCA coordinates (mean_temp) of the current class:
            dist = np.linalg.norm(new_cord_pca - mean_temp)
            distances += [dist]
        min = np.argmin(distances)

        #Temp Threshold
        threshold = 100000
        if distances[min] < threshold:
            print("Person", k, ":", min, self.folder_names[min])
            return self.folder_names[min]
        else:
            print("Person", k, ":", min, 'Unknown')
            return 'Unknown'
    
    
    def new_cord(self, img_path, img_height, img_width): #change name to path
        '''
        Returns new coordinates of your test image in the new bases. For you model to recognize your test image, it has to be in the same bases as your PCA
        '''
        img = cv2.imread(img_path)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        img_vec = np.asmatrix(gray).ravel() #flatten image
        img_vec = img_vec.T #take transpose
        new_mean = ((self.mean_face * len(self.y)) + img_vec)/(len(self.y) + 1) # find new mean after adding this image
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def original_data(self, new_coordinates):
        '''
            Gives you back the original data from new coordinates. Original image is
            nothing but the dot product of new bases and the transpose of coordinates
        '''
        return self.mean_face + (np.dot(self.new_bases, new_coordinates.T))


    def show_eigen_face(self, height, width, min_pix_int, max_pix_int, eig_no):
        '''
        eig_no: Specifies which eigenface to select from self.new_bases.
        min and max pixels intensity is the intensitry of pixels you want in grayscale
        '''

        #Retrieve the specific column vector representing the selected eigenface from the new coordinates
        ev = self.new_bases[:, eig_no:eig_no + 1]

        #Minimum and maximum values in the eigenface vector (ev).
        min_orig = np.min(ev)
        max_orig = np.max(ev)

        # Normalizes the eigenface values (ev) to fit within the specified intensity range.
        # or Adjust the eigenface values (ev) to fit within the specified intensity range (min_pix_int to max_pix_int)
        ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * ev)
       
        #reshape into 2D
        ev_re = np.reshape(ev, (height, width)) #de flattening the image
        
        #show in 200x200. Pause until a key is pressed
        cv2.imshow("Eigen Face " + str(eig_no),  cv2.resize(np.array(ev_re, dtype = np.uint8),(200, 200)))
        cv2.waitKey()

#show image - show_eigen_face