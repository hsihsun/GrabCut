import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy import misc
from scipy.stats import multivariate_normal  
from numpy import linalg as LA
import cv2
import matplotlib.pyplot as plt
import time
import math as ma
import maxflow
import sys

class Grabcut:

	def __init__(self, img, no_iter, no_GMM, gamma, debug):
	    self.im = img.copy()	
	    self.im_user = img.copy()
	    self.no_iter = no_iter 
	    self.alpha = np.zeros((np.size(img,0), np.size(img,1)))	
	    self.F_seed = np.zeros((np.size(img,0), np.size(img,1)))	
	    self.B_seed = np.zeros((np.size(img,0), np.size(img,1)))	
	    self.rect = []
	    self.ROI  = []
	    self.background = True
	    self.drawing = False
	    self.seg = np.zeros((np.size(img,0), np.size(img,1), np.size(img,2)))
	    self.mask = np.zeros_like(img, np.uint8)
	    self.no_GMM = int(no_GMM)
	    self.gamma = float(gamma)
	    self.debug = int(debug)
		    
	def fit_gmm(self, data, label):
		pi, mean, cov=[], [], []	
		channels = np.size(data,1)
		
		for x in range(self.no_GMM):
			N_cluster=data[label==x,:]
			size=N_cluster.shape
			pi.append(N_cluster.shape[0]/data.shape[0]+1e-8)
			mean.append(np.mean(N_cluster,axis=0)+ np.ones((channels))*1e-8)
			
			#print(N_cluster)
			cov.append(np.cov(N_cluster,rowvar=0) + np.eye(channels)*1e-8)
		return pi, mean, cov
	

	def GMM_inference(self, T, T_gmm):
		no_gmm=np.size(T_gmm[0])
		[pi, mean, cov]=T_gmm
		Q = np.zeros((np.size(T,0),no_gmm))
		#print( mean[0])
		for x in range(no_gmm):
			#print(im1D.shape, mean[x].shape, cov[x].shape)
			Q[:,x] = pi[x]*multivariate_normal.pdf(T,mean[x],cov[x],allow_singular=True)	 
			#Q = ma.log(Q)+ma.log(pi[x])
		return np.argmax(Q, axis=1)

	def calculate_beta(self):
		[H, W, Channels] = np.shape(self.im)	
		cum, cnt = 0, 0
		for y in range(H):
			for x in range(W):
				if y == self.im.shape[0]-1 or x == self.im.shape[1]-1:
					continue
					
				point = self.im[y, x, :]
				neighbor = np.array([self.im[y, x+1, :], self.im[y+1, x, :]])
				cum += np.sum(np.square(LA.norm(np.tile(point, (2,1))-neighbor, axis=1)))
				cnt = cnt+2
		return 1.0/(2*(cum/cnt))

	def calculate_V(self, c1, L1, c2, L2, beta):
		'''print(c1, L1, c1, L1)
		print(c1, L1, c1, L1)
		print(np.square(LA.norm(c1-c2)), beta, np.exp(-beta*(np.square(LA.norm(c1-c2)))), float(L1!=L2))
		print(gamma*float(L1!=L2)*np.exp(-beta*(np.square(LA.norm(c1-c2)))))
		'''
		return self.gamma*np.exp(-beta*(np.square(LA.norm(c1-c2))))


	def data_term(self, T, T_gmm):
		no_gmm=np.size(T_gmm[0])
		[pi, mean, cov]=T_gmm
		Q = np.zeros((np.size(T,0),no_gmm))	

		for x in range(no_gmm):
			Q[:,x] += pi[x]*multivariate_normal.pdf(T,mean[x],cov[x],allow_singular=True)

			#print(Q[:,x])
			'''if Q[:,x] == 0:
				Q[:,x] = 1e-15	''' 
			Q[Q[:,x] == 0,x] = 1e-15
			Q[:,x] = -np.log(Q[:,x])
		return np.amin(Q, axis=1)    
	
	def Cut(self):
		#print(self.im.shape)
		#print(H, W, channels)
		# select ROI
		#ROI = self.im[ymin:ymax,xmin:xmax,:]
		im1D = np.resize(self.im, (H*W,channels))
		#alpha = np.zeros((H,W))
		#cv2.imshow('alpha',self.alpha)
		alpha1D = np.resize(self.alpha, (H*W))
		#print(np.size(alpha,0), np.size(alpha ,1))

		TU=im1D[alpha1D==1,:]
		TB=im1D[alpha1D==0,:]

		#print(TU.shape, TB.shape)

		#self.no_GMM=5

		TU_label = KMeans(n_clusters=self.no_GMM, max_iter=10, random_state=None).fit_predict(TU)
		TB_label = KMeans(n_clusters=self.no_GMM, max_iter=10, random_state=None).fit_predict(TB)

		'''#print(TU_label.size, TB_label.size)	
		TUT = np.zeros((np.size(self.im,0), np.size(self.im,1), 3))
		TUT[:,:,0] = np.resize(im1D[:,0], (np.size(self.im,0), np.size(self.im,1)))/255
		TUT[:,:,1] = np.resize(im1D[:,1], (np.size(self.im,0), np.size(self.im,1)))/255
		TUT[:,:,2] = np.resize(im1D[:,2], (np.size(self.im,0), np.size(self.im,1)))/255
		plt.imshow(TUT)
		plt.show()'''

		TU_gmm = self.fit_gmm(TU, TU_label)
		TB_gmm = self.fit_gmm(TB, TB_label)

		#print(TU_gmm[0], TB_gmm[0])	

		start = time.time()

		for ite in range(int(self.no_iter)):		
			TU_label = KMeans(n_clusters=self.no_GMM, max_iter=10, random_state=None).fit_predict(TU)
			TB_label = KMeans(n_clusters=self.no_GMM, max_iter=10, random_state=None).fit_predict(TB)
			
			TU_gmm = self.fit_gmm(TU, TU_label)
			TB_gmm = self.fit_gmm(TB, TB_label)
			print('Iteration number: ', ite+1, '/', self.no_iter )

			TU_label=self.GMM_inference(TU, TU_gmm)    # calcuate the which label is the pixel belong to size(H*W)
			TB_label=self.GMM_inference(TB, TB_gmm)    # calcuate the which label is the pixel belong to size(H*W)

			TU_gmm=self.fit_gmm(TU, TU_label)   # calcuate the the gmm parameters for each lable {size(5), size(5)[3], size(5)[3, 3]}
			TB_gmm=self.fit_gmm(TB, TB_label)   # calcuate the the gmm parameters for each lable {size(5), size(5)[3], size(5)[3, 3]}

			'''TU_I = np.resize(TU_label, (np.size(self.ROI,0),np.size(self.ROI,1)))

			seg_TU = color.label2rgb(TU_I)

			cv2.imshow('seg_TU_after_EM', seg_TU )
			cv2.waitKey(0)
			cv2.destroyAllWindows()'''

			#self.gamma=0

			U_F = np.zeros(H*W)
			U_B = np.zeros(H*W)

			nodes = []
			edges = []

			F, B = np.zeros((H*W)), np.zeros((H*W))
			
			F = self.data_term(im1D, TU_gmm)
			B = self.data_term(im1D, TB_gmm)
			
			#start = time.time()

			for y in range(H):
				for x in range(W):
					point  = self.im[y,x,:]
					idx = y*W+x
					if y<self.rect[0] or y>self.rect[1] or x<self.rect[2] or x>self.rect[3]:
						nodes.append((idx, 1e10, 0))
					elif self.mask[y,x,2] == 255:
						nodes.append((idx, 0, 1e10))					
					elif self.mask[y,x,0] == 255:
						nodes.append((idx, 1e10, 0))
					else:
						#print('F', F[idx], 'B', B[idx])
						nodes.append((idx, F[idx], B[idx]))
			#end= time.time()
			#print (end - start)

			beta = self.calculate_beta()

			#start = time.time()
			
			test11 = np.zeros((H, W))

			for (y, x, c), value in np.ndenumerate(self.im):
				point = self.im[y,x,:]

				if y == self.im.shape[0]-1 or x == self.im.shape[1]-1:
					continue

				cur_idx = y*W+x
				nbr_idx = y*W+x+1
							 
				v = self.calculate_V(im1D[cur_idx], alpha1D[cur_idx], im1D[nbr_idx], alpha1D[nbr_idx], beta)
				#print('x+1', v)
				edges.append((cur_idx, nbr_idx, v))

				test11[y][x] = v
				
				nbr_idx = (y+1)*W+x
				
				v = self.calculate_V(im1D[cur_idx], alpha1D[cur_idx], im1D[nbr_idx], alpha1D[nbr_idx], beta)
				#print('y+1', v)
				edges.append((cur_idx, nbr_idx, v))	
				
				'''nbr_idx = (y+1)*W+x+1
				
				v = self.calculate_V(im1D[cur_idx], alpha1D[cur_idx], im1D[nbr_idx], alpha1D[nbr_idx], gamma, beta)

				edges.append((cur_idx, nbr_idx, v))
				
				if y != 0 and x != 0:

					nbr_idx = (y+1)*W+x-1
				
					v = self.calculate_V(im1D[cur_idx], alpha1D[cur_idx], im1D[nbr_idx], alpha1D[nbr_idx], gamma, beta)

					edges.append((cur_idx, nbr_idx, v))		'''
			
			'''cv2.imshow('test',test11)
			cv2.waitKey(0)
			cv2.destroyAllWindows()		'''	

			g = maxflow.Graph[float](len(nodes), len(edges))
			nlist = g.add_nodes(len(nodes))

			for n in nodes:
				g.add_tedge(nlist[n[0]], n[1], n[2])

			for e in edges:
				g.add_edge(e[0], e[1], e[2], e[2])

			flow = g.maxflow()
			alpha_prev = np.resize(alpha1D, (H,W))

			for idx in range(len(nodes)):
				if g.get_segment(idx) == 1:
					alpha1D[idx] = 1
				else: 
					alpha1D[idx] = 0

			TU=im1D[alpha1D==1,:]
			TB=im1D[alpha1D==0,:]

			self.alpha = np.resize(alpha1D, (H,W))	
			if self.debug == 1:
				print(100)
				cv2.imshow('alpha2D',self.alpha)
				cv2.waitKey(500)
				#cv2.destroyAllWindows()
		
			error = np.mean(np.abs(self.alpha-alpha_prev))
			print('error', error)
			
			if error < 1e-5:
				break
				
			

		end= time.time()
		print ('Computation time: ', end - start, ' second')
		
	def select_foreground(self):
		#global background 	
		self.background = False
	
	def select_background(self):
		#global background 
		self.background = True
			
	def draw_circle(self, event, x, y, flags, param):			
		#global drawing
		
		if self.background == True:
		    sketchColor = (255,0,0)
		else:	
		    sketchColor = (0,0,255)
	
		if event == cv2.EVENT_LBUTTONDOWN:	
			self.drawing = True
			ix, iy = x, y
		    
		elif event == cv2.EVENT_MOUSEMOVE:  	
			if self.drawing == True:
				cv2.circle(self.mask, (x, y), 5, sketchColor, -1)	
				cv2.circle(self.im_user, (x, y), 5, sketchColor, -1)		
			ix, iy = x, y		
						    
		elif event == cv2.EVENT_LBUTTONUP:	
			self.drawing = False
			cv2.circle(self.mask, (x, y), 5, sketchColor, -1)	
			cv2.circle(self.im_user, (x, y), 5, sketchColor, -1)				
			ix, iy = x, y				

	def run(self):	
		self.rect = cv2.selectROI(self.im_user)		
		rect = self.rect	
		cv2.destroyAllWindows()
		
		#print(r)
		H = np.size(self.im,0)
		W = np.size(self.im,1)
		ymin = max(min(rect[1],rect[1]+rect[3]),0)
		xmin = max(min(rect[0],rect[0]+rect[2]),0)
		ymax = min(max(rect[1],rect[1]+rect[3]),H-1)
		xmax = min(max(rect[0],rect[0]+rect[2]),W-1)
		self.rect = [ymin, ymax, xmin, xmax]
		self.alpha[ymin:ymax,xmin:xmax] = 1
		self.ROI = self.im[ymin:ymax,xmin:xmax]
		print(xmin, xmax, ymin, ymax)	
		
		cv2.imshow('alpha2D',self.alpha)
		cv2.waitKey(500)
		
		self.Cut()	

		#cv2.imshow('alpha2D',self.alpha)
		alpha = np.repeat(self.alpha[:,:,np.newaxis], 3, axis=2)
		
		#seg = np.zeros((H,W,channels))
		
		for c in range(channels):
			self.seg[:,:,c] = self.im[:,:,c]*alpha[:,:,c]/255
			
		cv2.imshow('Segmentation result',self.seg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
		# self.no_iter = 5
		# global 	drawing
		self.drawing = False
		print(' Button: f->foreground, b->background, n-> reset, r->run next grabcut, q->termination and show final result')		
		self.im_user
		
		while(1):					
			cv2.namedWindow('User Interaction')	
			cv2.setMouseCallback('User Interaction',self.draw_circle)		
			cv2.imshow('User Interaction',self.im_user)	
			k = cv2.waitKey(1) & 0xFF	
	
			if k == ord('f'):
				self.select_foreground()
				
			if k == ord('b'):
				self.select_background()
				
			if k == ord('n'):
				self.im_user = self.im.copy()
				self.mask = np.zeros_like(self.im_user, np.uint8)
				
			if k == ord('r'):		
				cv2.destroyAllWindows()
				cv2.imshow('alpha2D',self.alpha)
				cv2.waitKey(500)
				
				self.Cut()
				
				cv2.imshow('alpha2D',self.alpha)		
				alpha = np.repeat(self.alpha[:,:,np.newaxis], 3, axis=2)
				seg = np.zeros((H,W,channels))
				for c in range(channels):
					self.seg[:,:,c] = self.im[:,:,c]*alpha[:,:,c]/255
			
				cv2.imshow('Segmentation result',self.seg)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				
				self.im_user = self.im.copy()
															
			if k == ord('q'):
				break
	
		alpha = np.repeat(self.alpha[:,:,np.newaxis], 3, axis=2)
		
		#seg = np.zeros((H,W,channels))
		
		for c in range(channels):
			self.seg[:,:,c] = self.im[:,:,c]*alpha[:,:,c]/255
			
		cv2.imshow('Final segmentation result',self.seg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		print('Grabcut is finished')

if __name__ == '__main__':
	
	imagePath=sys.argv[1]	
	no_iter=sys.argv[2]	
	no_GMM=sys.argv[3]	
	d=sys.argv[4]	
	gamma = sys.argv[5]
	debug = sys.argv[6]
		
	I = cv2.imread(imagePath, cv2.IMREAD_COLOR)

	[H, W, channels] = I.shape
	im2 = cv2.resize(I, (int(W/float(d)), int(H/float(d))))	
	
	[H, W, channels] = im2.shape
	print('Image scale: Height:', H, ' Width:', W, ' Channels', channels)
	
	Gb = Grabcut(im2, no_iter, no_GMM, gamma, debug)
	Gb.run()

















