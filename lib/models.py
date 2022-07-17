"""

PYTORCH models for use in this project.

"""

from .utils import *
from .nn_utils import *
from .gp_kernel import *
import os
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal, kl_divergence
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


# Encoders
class Decoder(nn.Module):


	def __init__(self, z_dim=32,x_dim=128*128,decoder_dist = False):
		"""
		Initialize stupid decoder

		Inputs
		-----
			z_dim: int, dim of latent dimension
			x_dim: int, dim of input data
			decoder_dist: bool, determines if we learn var of decoder in addition
						to mean
		"""

		super(Decoder,self).__init__()
		self.decoder_dist = decoder_dist

		self.decoder_fc = nn.Sequential(nn.Linear(z_dim,64),
										nn.Linear(64,256),
										nn.Linear(256,1024),
										nn.Softplus(),
										nn.Linear(1024,8192),
										nn.Softplus())
		self.decoder_convt = nn.Sequential(nn.BatchNorm2d(32),
										nn.ConvTranspose2d(32,24,3,1,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(24),
										nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(24),
										nn.ConvTranspose2d(24,16,3,1,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(16),
										nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(16),
										nn.ConvTranspose2d(16,8,3,1,padding=1),
										nn.Softplus())

		self.mu_convt = nn.Sequential(nn.BatchNorm2d(8),
									nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1),
									nn.Softplus(),
									nn.BatchNorm2d(8),
									nn.ConvTranspose2d(8,1,3,1,padding=1))
		if self.decoder_dist:
			self.min_logvar = torch.ones(x_dim,dtype=torch.float64)*(-6)
			self.logvar_convt = nn.Sequential(nn.BatchNorm2d(8),
										nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(8),
										nn.ConvTranspose2d(8,1,3,1,padding=1))

		device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)

		self.to(self.device)


	def forward(self,z):
		"""
		Decode latent samples

		Inputs
		-----
			z: torch.tensor, latent samples to be decoded

		Outputs
		-----
			mu: torch.tensor, mean of decoded distribution
			if decoder_dist:
				logvar: torch.tensor, logvar of decoded distribution
		"""
		#print(z.dtype)
		z = self.decoder_fc(z)
		z = z.view(-1,32,16,16)
		z = self.decoder_convt(z)
		mu = self.mu_convt(z)

		if self.decoder_dist:
			logvar = self.logvar_convt(z)
			logvar = self.min_logvar + nn.ReLU(logvar - self.min_logvar)
			return (mu,logvar)

		else:
			return mu

	def draw_sample(self,mu,logvar):

		"""
		Draw sample from decoder distribution according to
		given mean and logvar

		Inputs:
		-----
			mu: torch.tensor, mean of decoder distribution
			logvar: torch.tensor, logvar of decoder distribution

		Outputs:
		-----
			sample: torch.tensor, sample from decoder distribution
		"""

		sample = torch.randn(mu.shape,dtype=torch.float64)
		sample = mu + sample * torch.exp(0.5 * logvar)

		return sample

class Encoder(nn.Module):

	"""VAE Encoder for latent SDE bird experiments. because apparently everyone
	fucking writes their code like this. fuck you.
	"""
	def __init__(self,latent_dim=32):

		"""
		Initalizes encoder. Setup is same as from Jack's VAE experiments

		Inputs
		-----
			latent_dim: int, number of latent dims to encode to
		"""
		super(Encoder,self).__init__()

		self.min_logvar = torch.ones(latent_dim,dtype=torch.float64)*(-6)
		self.latent_dim = latent_dim
		self.encoder_conv = nn.Sequential(nn.BatchNorm2d(1),
										nn.Conv2d(1, 8, 3,1,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(8),
										nn.Conv2d(8, 8, 3,2,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(8),
										nn.Conv2d(8, 16,3,1,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(16),
										nn.Conv2d(16,16,3,2,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(16),
										nn.Conv2d(16,24,3,1,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(24),
										nn.Conv2d(24,24,3,2,padding=1),
										nn.Softplus(),
										nn.BatchNorm2d(24),
										nn.Conv2d(24,32,3,1,padding=1),
										nn.Softplus())

		self.encoder_fc = nn.Sequential(nn.Linear(8192,1024),
										nn.Softplus(),
										nn.Linear(1024,256),
										nn.Softplus())
		self.encoder_mu = nn.Sequential(nn.Linear(256,64),
										nn.Softplus(),
										nn.Linear(64,self.latent_dim))
		self.encoder_u = nn.Sequential(nn.Linear(256,64),
										nn.Softplus(),
										nn.Linear(64,self.latent_dim))
		self.encoder_d = nn.Sequential(nn.Linear(256,64),
										nn.Softplus(),
										nn.Linear(64,self.latent_dim))

		device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		self.min_logvar = self.min_logvar.to(self.device)

		self.to(self.device)


	def forward(self,x):

		"""
		Encode data - setup from my own dang brain
		this is for a batch (vocalization)
	

		Inputs
		-----
			x: torch.tensor, data to be encoded

		Outputs
		-----
			mu: torch.tensor, encoder dist. mean
			u: torch.tensor, encoder dist. cholesky decomp
			d: torch.tensor, encoder dist. diagonal
		"""

		# output shapes:
		# mu: T x latent_dim
		# us: T x latent_dim
		# ds: T x latent_dim
		x = self.encoder_conv(x)
		x = x.view(-1, 8192)
		x = self.encoder_fc(x)
		mus = self.encoder_mu(x)
		#mus = [nn.Softplus()(fc(mu)) for fc in self.fc_m2]
		us = self.encoder_u(x)
		ds = self.encoder_d(x)
		#precs = [nn.Softplus()(fc(prec)) for prec in self.fc_c2] 
		#log_var = self.fc22(self.fc12(x))

		#self.min_logvar.to(self.device)
		#print(self.min_logvar.device)
		#print(log_var.device)
		#log_var = self.min_logvar + nn.ReLU()(log_var - self.min_logvar)

		# transpose each: pytorch will now treat latent dim as batch, T as dim of data
		us = torch.zeros(us.shape,device=self.device,dtype=torch.float32)
		return torch.transpose(mus,0,1), torch.transpose(us,0,1).unsqueeze(-1), torch.transpose(ds,0,1).exp()

	def draw_samples(self, z_mean, z_u, z_d):

		"""
		Draw a single sample from encoder distribution. need an entire batch (or subbatch)
		to draw samples, given that all samples are linked

		Inputs
		-----
			z_mean: torch.tensor, dist. mean

			z_u: covariance factor of dist.

			z_d: diagonal of dist. covar.

		Outputs
		-----
			sample: torch.tensor, sample from dist.
		"""
		
		### This assumes input is in shape: latent_dim x T!! 
		### this way torch will treat dim as independent, 
		dists = LowRankMultivariateNormal(z_mean,z_u,z_d)
		samples = dists.sample()
		#samples = torch.stack(samples, axis=-1)
		#sample = torch.randn(z_mean.shape,dtype=torch.float64)
		#sample = z_mean + sample * torch.exp(0.5 * z_logvar)

		return samples

class GPVAE(nn.Module):

	def __init__(self, encoder, decoder,save_dir,lr=1e-4,plots_dir='',precision=10,
				kernel='cauchy',length_scale=4,beta=0.5):


		"""
		Initialize full VAE for ODE experiments. Why this needs to be separate
		from the encoder and decoder: ????
		however, this is what people do, so I will do it.

		Inputs
		-----
			latent_dim: int, latent dimension
			x_dim: int, dimension of data
			encoder_path: string, path for saving encoder
			decoder_path: string, path for saving decoder
			decoder_dist: bool, whether or not we want the encoder to return a
						learned distribution or one with a pre-specified variance
		"""
		super(GPVAE,self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.latent_dim = self.encoder.latent_dim
		#self.x_dim = x_dim
		#self.decoder_dist = decoder_dist
		self.save_dir = save_dir
		if not os.path.isdir(plots_dir):
			os.mkdir(plots_dir)
		self.plots_dir = plots_dir 
		self.epoch = 0
		self.lr = lr
		self.loss = {'train': {}, 'test': {}}

		### maybe change later
		self.M = 1
		self.K = 1
		self.beta=beta
		self.kernel_scales=1
		self.length_scale=length_scale
		self.sigma=1.005
		self.kernel = kernel
		self.precision=precision
		#if device_name == "auto":
		device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		self.to(self.device)
		self.optimizer = Adam(self.parameters(), lr=lr) # 8e-5

	def set_epoch(self,epoch):
		"""
		sets current epoch. mainly just used for saving checkpoints.
		"""

		self.previous_epochs = epoch

	def predict_decoder(self,z):

		"""
		Decode data! If learned distribution, sample from decoded distribution

		Inputs
		-----
			z: torch.tensor, data to be decoded

		Outputs:
		-----
			y: torch.tensor, data decoded from latent space
		"""

		if self.decoder_dist:
			(mu,logvar) = self.decoder.forward(z)
			y = self.decoder.draw_sample(mu,logvar)

		else:
			y = self.decoder.forward(z)

		return y

	def save_state(self):

		"""
		Save state of network. Saves encoder and decoder state separately. Assumes
		that while training, you have been using set_epoch to set the epoch
		"""

		#self.set_epoch(epoch)
		fn = os.path.join(self.save_dir, 'checkpoint_encoder_' + str(self.epoch) + '.tar')

		"""Save state."""
		sd = self.state_dict()
		torch.save({
				'model_state_dict': sd,
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': self.epoch
			}, fn)

	def load_state(self,fn):

		"""
		Load state of network. Requires an epoch to recover the current state of network

		Inputs:
		-----
			epoch: int, current epoch of training
		"""
		"""Load state."""

		print("Loading state from:", fn)
		#print(self.state_dict().keys())

		checkpoint = torch.load(fn, map_location=self.device)
		#layer_1 = checkpoint['model_state_dict'].pop('layer_1')

		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.epoch = checkpoint['epoch']
		
	def generate(self, noise=None, num_samples=1):
		if noise is None:
			noise = torch.randn((num_samples, self.latent_dim))
		return self.decoder.decode(noise)
	
	def _get_prior(self,T):
		
		# Compute kernel matrices for each latent dimension
		kernel_matrices = []
		for i in range(self.kernel_scales):
			if self.kernel == "rbf":
				kernel_matrices.append(rbf_kernel(T, int(T/6) / 2**i))
			elif self.kernel == "diffusion":
				kernel_matrices.append(diffusion_kernel(T, int(T/6) / 2**i))
			elif self.kernel == "matern":
				kernel_matrices.append(matern_kernel(T, T / 2**i))
			elif self.kernel == "cauchy":
				kernel_matrices.append(cauchy_kernel(T, self.sigma,T*2 / 2**i))
			else:
				raise NotImplementedError

		# Combine kernel matrices for each latent dimension
		tiled_matrices = []
		total = 0
		for i in range(self.kernel_scales):
			if i == self.kernel_scales-1:
				multiplier = self.latent_dim - total
			else:
				multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
				total += multiplier
			tiled_matrices.append(torch.tile(torch.unsqueeze(kernel_matrices[i], 0), [multiplier, 1, 1]))
		kernel_matrix_tiled = torch.tensor(np.concatenate(tiled_matrices),device=self.device)
		assert len(kernel_matrix_tiled) == self.latent_dim

		self.prior = MultivariateNormal(
			loc=torch.zeros([self.latent_dim, T], dtype=torch.float32,device=self.device),
			covariance_matrix=kernel_matrix_tiled)
		return self.prior

	def compute_nll(self, x, y=None, m_mask=None):
		# Used only for evaluation
		assert len(x.shape) == 4, "Input should have shape: [time_length, n_chan, h,w]"
		if y is None: y = x

		mus,us,ds = self.encoder.encode(x)
		z_samples = self.encoder.draw_samples(mus,us,ds)
		# reshaping to be T x Z_dim
		x_hat = self.decoder.decode(torch.transpose(z_samples,0,1))
		x_hat_dist = Normal(x_hat, 1/self.precision)
		nll = -x_hat_dist.log_prob(y)  # shape=(TL, D)
		nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
		if m_mask is not None:
			m_mask = m_mask.to(torch.bool)
			nll = torch.where(m_mask, nll, torch.zeros_like(nll))  # !!! inverse mask, set zeros for observed
		return torch.sum(nll)

	def compute_mse(self, x, y=None, m_mask=None, binary=False):
		# Used only for evaluation
		assert len(x.shape) == 4, "Input should have shape: [time_length,n_chan,h,w]"
		if y is None: y = x

		z_means,_,_ = self.encoder.encode(x)
		x_hat_mean = self.decoder.decode(torch.transpose(z_means,0,1))  # shape=(TL, h*w)
		if binary:
			x_hat_mean = torch.round(x_hat_mean)
		mse = torch.pow(x_hat_mean - y.view(y.shape[0],-1), 2)
		if m_mask is not None:
			m_mask = m_mask.to(torch.bool)
			mse = torch.where(m_mask, mse, torch.zeros_like(mse))  # !!! inverse mask, set zeros for observed
		return torch.sum(mse)

	def _compute_loss(self, x, m_mask=None, return_parts=False):
		assert len(x.shape) == 4, "Input should have shape: [time_length, n_chan,h,w]"
		#x = nn.Identity(x)  # in case x is not a Tensor already...
		T = x.shape[0]
		x = torch.tile(x, [self.M * self.K, 1,1, 1])  # shape=(M*K*T, n_chan, h,w)

		if m_mask is not None:
			#m_mask = nn.Identity(m_mask)  # in case m_mask is not a Tensor already...
			m_mask = torch.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
			m_mask = m_mask.to(torch.bool)

		pz = self._get_prior(T)
		mus,us,ds = self.encoder.forward(x)
		
		qz_x = LowRankMultivariateNormal(mus,us,ds)
		
		zhat = torch.transpose(qz_x.sample(),0,1)
		xhat = self.decoder.forward(zhat)

		px_z = Normal(xhat, 1/self.precision)

		nll = -px_z.log_prob(x)  # shape=(M*K*TL, D)
		nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
		if m_mask is not None:
			nll = torch.where(m_mask, torch.zeros_like(nll), nll)  # if not HI-VAE, m_mask is always zeros
		nll = torch.sum(nll, dim=-1)  # shape=(M*K*BS)

		if self.K > 1:
			kl = qz_x.log_prob(zhat) - pz.log_prob(zhat)  # shape=(M*K*TL, or d)
			kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
			kl = torch.sum(kl, 1)  # shape=(M*K*BS)

			weights = -nll - kl  # shape=(M*K*BS)
			weights = torch.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, T)

			elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
			elbo = torch.mean(elbo)  # scalar
		else:
			# if K==1, compute KL analytically
			kl = kl_divergence(qz_x, pz)  # shape=(z_dim)
			#print(kl.shape)
			kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
			kl = torch.sum(kl)  # shape=(M*K*BS)

			elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1
			elbo = torch.mean(elbo)  # scalar

		if return_parts:
			nll = torch.mean(nll)  # scalar
			kl = torch.mean(kl)  # scalar
			return -elbo, nll, kl, xhat.view(-1,128,128).detach().cpu().numpy()
		else:
			return -elbo

	'''
	def compute_loss(self, x, m_mask=None, return_parts=False):
		del m_mask
		return self._compute_loss(x, return_parts=return_parts)
	'''
	def train_epoch(self,loader):

		self.train()
		### each batch is one trajectory. 
		train_loss = 0.0
		train_nlp = 0.0
		train_kl = 0.0
		for ind, (spec,day) in enumerate(loader):

			self.optimizer.zero_grad()
			day = day.to(self.device).squeeze()
			#print(day.shape)
			#print(spec.shape)
			#spec = torch.stack(spec,axis=0)
			spec = torch.stack(spec).to(self.device).squeeze().unsqueeze(1)

			loss,nll,kl,_ = self._compute_loss(spec,return_parts=True)

			loss.backward()

			self.optimizer.step()

			train_loss += loss.item()
			train_nlp += nll.item()
			train_kl += kl.item()
		
		train_loss /= len(loader)
		train_nlp /= len(loader)
		train_kl /= len(loader)

		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))
		print('Epoch {0:d} average train nll: {1:.3f}'.format(self.epoch,train_nlp))
		print('Epoch {0:d} average train kl: {1:.3f}'.format(self.epoch,train_kl))
		
		return train_loss

	def test_epoch(self,loader):

		self.eval()

		test_loss = 0.0
		test_nlp = 0.0
		test_kl = 0.0
		for ind, (spec,day) in enumerate(loader):

			day = day.to(self.device).squeeze()
			#print(day.shape)
			#print(spec.shape)
			#spec = torch.stack(spec,axis=0)
			spec = torch.stack(spec).to(self.device).squeeze().unsqueeze(1)
			with torch.no_grad():
				loss,nll,kl,_ = self._compute_loss(spec,return_parts=True)

			

			test_loss += loss.item()
			test_nlp += nll.item()
			test_kl += kl.item()
		
		test_loss /= len(loader)
		test_nlp /= len(loader)
		test_kl /= len(loader)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))
		print('Epoch {0:d} average test nll: {1:.3f}'.format(self.epoch,test_nlp))
		print('Epoch {0:d} average test kl: {1:.3f}'.format(self.epoch,test_kl))

		return test_loss

	def train_test_loop(self,loaders, epochs=100, test_freq=2, save_freq=10,
		vis_freq=1):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
			torch.utils.data.Dataloader objects.
		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to ``100``.
		test_freq : int, optional
			Testing is performed every `test_freq` epochs. Defaults to ``2``.
		save_freq : int, optional
			The model is saved every `save_freq` epochs. Defaults to ``10``.
		vis_freq : int, optional
			Syllable reconstructions are plotted every `vis_freq` epochs.
			Defaults to ``1``.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Training set:", len(loaders['train'].dataset))
		print("Test set:", len(loaders['test'].dataset))
		print("="*40)
		# For some number of epochs...
		for epoch in range(self.epoch, self.epoch+epochs):
			# Run through the training data and record a loss.
			loss = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			# Run through the test data and record a loss.
			if (test_freq is not None) and (epoch % test_freq == 0):
				loss = self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = loss
			# Save the model.
			if (save_freq is not None) and (epoch % save_freq == 0) and \
					(epoch > 0):
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state()
			# Plot reconstructions.
			if (vis_freq is not None) and (epoch % vis_freq == 0):
				self.visualize(loaders['test'])

			self.epoch += 1

	def visualize(self, loader, num_specs=5):
		"""
		Plot spectrograms and their reconstructions.

		Spectrograms are chosen at random from the Dataloader Dataset.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			Spectrogram Dataloader
		num_specs : int, optional
			Number of spectrogram pairs to plot. Defaults to ``5``.
		gap : int or tuple of two ints, optional
			The vertical and horizontal gap between images, in pixels. Defaults
			to ``(2,6)``.
		save_filename : str, optional
			Where to save the plot, relative to `self.save_dir`. Defaults to
			``'temp.pdf'``.

		Returns
		-------
		specs : numpy.ndarray
			Spectgorams from `loader`.
		rec_specs : numpy.ndarray
			Corresponding spectrogram reconstructions.
		"""
		# Collect random indices.
		assert num_specs <= len(loader.dataset) and num_specs >= 1
		indices = np.random.choice(np.arange(len(loader.dataset)),
			size=num_specs,replace=False)
		
		(specs,days) = loader.dataset[indices]
		for spec in specs:
			spec = torch.stack(spec).to(self.device).squeeze().unsqueeze(1)


			# Retrieve spectrograms from the loader.
			# Get resonstructions.
			with torch.no_grad():
				_, _, _,rec_specs = self._compute_loss(spec, return_parts=True)
			spec = spec.detach().cpu().numpy()
			nrows = 1 + spec.shape[0]//5
			fig,axs = plt.subplots(nrows=nrows,ncols=5)
			row_ind = 0
			col_ind = 0

			for im in range(rec_specs.shape[0]):

				if col_ind >= 5:
					row_ind += 1
					col_ind = 0

				axs[row_ind,col_ind].imshow(rec_specs[im,:,:],origin='lower')
				axs[row_ind,col_ind].get_xaxis().set_visible(False)
				axs[row_ind,col_ind].get_yaxis().set_visible(False)
				col_ind += 1

			for ii in range(col_ind,5):
				axs[row_ind,ii].get_xaxis().set_visible(False)
				axs[row_ind,ii].get_yaxis().set_visible(False)
				axs[row_ind,ii].axis('square')


			#all_specs = np.stack([specs, rec_specs])
		# Plot.
			save_fn = 'reconstruction_epoch_' + str(self.epoch) + '_' + str(im) + '.png' 
			save_filename = os.path.join(self.plots_dir, save_fn)

			plt.savefig(save_filename)
			plt.close('all')

			fig,axs = plt.subplots(nrows=nrows,ncols=5)
			row_ind = 0
			col_ind = 0

			for im in range(spec.shape[0]):

				if col_ind >= 5:
					row_ind += 1
					col_ind = 0

				axs[row_ind,col_ind].imshow(spec[im,:,:,:].squeeze(),origin='lower')
				axs[row_ind,col_ind].get_xaxis().set_visible(False)
				axs[row_ind,col_ind].get_yaxis().set_visible(False)
				col_ind += 1

			for ii in range(col_ind,5):
				axs[row_ind,ii].get_xaxis().set_visible(False)
				axs[row_ind,ii].get_yaxis().set_visible(False)
				axs[row_ind,ii].axis('square')

			#all_specs = np.stack([specs, rec_specs])
		# Plot.
			save_fn = 'real_epoch_' + str(self.epoch) + '_' + str(im) + '.png' 
			save_filename = os.path.join(self.plots_dir, save_fn)

			plt.savefig(save_filename)
			plt.close('all')

		return 
	'''
	MAYBE return to this when everything works. MAYBE.
	def kl_divergence(self, a, b):
		""" Batched KL divergence `KL(a || b)` for multivariate Normals.
			See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
					   /python/distributions/mvn_linear_operator.py
			It's used instead of default KL class in order to exploit precomputed components for efficiency
		"""

		def squared_frobenius_norm(x):
			"""Helper to make KL calculation slightly more readable."""
			return torch.sum(torch.pow(x,2), dim=(-2, -1))

		def is_diagonal(x):
			"""Helper to identify if `LinearOperator` has only a diagonal component."""
			return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
					isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
					isinstance(x, tf.linalg.LinearOperatorDiag))

		if is_diagonal(a.scale) and is_diagonal(b.scale):
			# Using `stddev` because it handles expansion of Identity cases.
			b_inv_a = (a.stddev() / b.stddev())[..., None]
		else:
			if self.pz_scale_inv is None:
				self.pz_scale_inv = torch.linalg.inv(b.scale.to_dense())
				self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
											 self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

			if self.pz_scale_log_abs_determinant is None:
				self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

			a_shape = a.scale.shape
			if len(b.scale.shape) == 3:
				_b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
			else:
				_b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

			b_inv_a = _b_scale_inv @ a.scale.to_dense()

		# ~10x times faster on CPU then on GPU
		with tf.device('/cpu:0'):
			kl_div = (self.pz_scale_log_abs_determinant - a.scale.log_abs_determinant() +
					  0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
					  squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
					  b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
		return kl_div
	'''
	#def kl_divergence(self, a, b):
	#	return tfd.kl_divergence(a, b)

	#def get_trainable_vars(self):
	#	self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
	#					  tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
	#	return self.trainable_variables

'''
Things I may not need! encoders from Fortuin paper
class DiagonalEncoder(tf.keras.Model):
	def __init__(self, z_size, hidden_sizes=(64, 64), **kwargs):
		""" Encoder with factorized Normal posterior over temporal dimension
			Used by disjoint VAE and HI-VAE with Standard Normal prior
			:param z_size: latent space dimensionality
			:param hidden_sizes: tuple of hidden layer sizes.
								 The tuple length sets the number of hidden layers.
		"""
		super(DiagonalEncoder, self).__init__()
		self.z_size = int(z_size)
		self.net = make_nn(2*z_size, hidden_sizes)

	def __call__(self, x):
		mapped = self.net(x)
		return tfd.MultivariateNormalDiag(
		  loc=mapped[..., :self.z_size],
		  scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))


class JointEncoder(tf.keras.Model):
	def __init__(self, z_size, hidden_sizes=(64, 64), window_size=3, transpose=False, **kwargs):
		""" Encoder with 1d-convolutional network and factorized Normal posterior
			Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
			:param z_size: latent space dimensionality
			:param hidden_sizes: tuple of hidden layer sizes.
								 The tuple length sets the number of hidden layers.
			:param window_size: kernel size for Conv1D layer
			:param transpose: True for GP prior | False for Standard Normal prior
		"""
		super(JointEncoder, self).__init__()
		self.z_size = int(z_size)
		self.net = make_cnn(2*z_size, hidden_sizes, window_size)
		self.transpose = transpose

	def __call__(self, x):
		mapped = self.net(x)
		if self.transpose:
			num_dim = len(x.shape.as_list())
			perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
			mapped = tf.transpose(mapped, perm=perm)
			return tfd.MultivariateNormalDiag(
					loc=mapped[..., :self.z_size, :],
					scale_diag=tf.nn.softplus(mapped[..., self.z_size:, :]))
		return tfd.MultivariateNormalDiag(
					loc=mapped[..., :self.z_size],
					scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))


class BandedJointEncoder(tf.keras.Model):
	def __init__(self, z_size, hidden_sizes=(64, 64), window_size=3, data_type=None, **kwargs):
		""" Encoder with 1d-convolutional network and multivariate Normal posterior
			Used by GP-VAE with proposed banded covariance matrix
			:param z_size: latent space dimensionality
			:param hidden_sizes: tuple of hidden layer sizes.
								 The tuple length sets the number of hidden layers.
			:param window_size: kernel size for Conv1D layer
			:param data_type: needed for some data specific modifications, e.g:
				tf.nn.softplus is a more common and correct choice, however
				tf.nn.sigmoid provides more stable performance on Physionet dataset
		"""
		super(BandedJointEncoder, self).__init__()
		self.z_size = int(z_size)
		self.net = make_cnn(3*z_size, hidden_sizes, window_size)
		self.data_type = data_type

	def __call__(self, x):
		mapped = self.net(x)

		batch_size = mapped.shape.as_list()[0]
		time_length = mapped.shape.as_list()[1]

		# Obtain mean and precision matrix components
		num_dim = len(mapped.shape.as_list())
		perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
		mapped_transposed = tf.transpose(mapped, perm=perm)
		mapped_mean = mapped_transposed[:, :self.z_size]
		mapped_covar = mapped_transposed[:, self.z_size:]

		# tf.nn.sigmoid provides more stable performance on Physionet dataset
		if self.data_type == 'physionet':
			mapped_covar = tf.nn.sigmoid(mapped_covar)
		else:
			mapped_covar = tf.nn.softplus(mapped_covar)

		mapped_reshaped = tf.reshape(mapped_covar, [batch_size, self.z_size, 2*time_length])

		dense_shape = [batch_size, self.z_size, time_length, time_length]
		idxs_1 = np.repeat(np.arange(batch_size), self.z_size*(2*time_length-1))
		idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2*time_length-1)), batch_size)
		idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_size)
		idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_size)
		idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

		# ~10x times faster on CPU then on GPU
		with tf.device('/cpu:0'):
			# Obtain covariance matrix from precision one
			mapped_values = tf.reshape(mapped_reshaped[:, :, :-1], [-1])
			prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
			prec_sparse = tf.sparse.reorder(prec_sparse)
			prec_tril = tf.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
			eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
			prec_tril = prec_tril + eye
			cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
			cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

		num_dim = len(cov_tril.shape)
		perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
		cov_tril_lower = tf.transpose(cov_tril, perm=perm)
		z_dist = tfd.MultivariateNormalTriL(loc=mapped_mean, scale_tril=cov_tril_lower)
		return z_dist


# Decoders

class Decoder(tf.keras.Model):
	def __init__(self, output_size, hidden_sizes=(64, 64)):
		""" Decoder parent class with no specified output distribution
			:param output_size: output dimensionality
			:param hidden_sizes: tuple of hidden layer sizes.
								 The tuple length sets the number of hidden layers.
		"""
		super(Decoder, self).__init__()
		self.net = make_nn(output_size, hidden_sizes)

	def __call__(self, x):
		pass


class BernoulliDecoder(Decoder):
	""" Decoder with Bernoulli output distribution (used for HMNIST) """
	def __call__(self, x):
		mapped = self.net(x)
		return tfd.Bernoulli(logits=mapped)


class GaussianDecoder(Decoder):
	""" Decoder with Gaussian output distribution (used for SPRITES and Physionet) """
	def __call__(self, x):
		mean = self.net(x)
		var = tf.ones(tf.shape(mean), dtype=tf.float32)
		return tfd.Normal(loc=mean, scale=var)


# Image preprocessor

class ImagePreprocessor(tf.keras.Model):
	def __init__(self, image_shape, hidden_sizes=(256, ), kernel_size=3.):
		""" Decoder parent class without specified output distribution
			:param image_shape: input image size
			:param hidden_sizes: tuple of hidden layer sizes.
								 The tuple length sets the number of hidden layers.
			:param kernel_size: kernel/filter width and height
		"""
		super(ImagePreprocessor, self).__init__()
		self.image_shape = image_shape
		self.net = make_2d_cnn(image_shape[-1], hidden_sizes, kernel_size)

	def __call__(self, x):
		return self.net(x)


# VAE models

class VAE(tf.keras.Model):
	def __init__(self, latent_dim, data_dim, time_length,
				 encoder_sizes=(64, 64), encoder=DiagonalEncoder,
				 decoder_sizes=(64, 64), decoder=BernoulliDecoder,
				 image_preprocessor=None, beta=1.0, M=1, K=1, **kwargs):
		""" Basic Variational Autoencoder with Standard Normal prior
			:param latent_dim: latent space dimensionality
			:param data_dim: original data dimensionality
			:param time_length: time series duration
			
			:param encoder_sizes: layer sizes for the encoder network
			:param encoder: encoder model class {Diagonal, Joint, BandedJoint}Encoder
			:param decoder_sizes: layer sizes for the decoder network
			:param decoder: decoder model class {Bernoulli, Gaussian}Decoder
			
			:param image_preprocessor: 2d-convolutional network used for image data preprocessing
			:param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
			:param M: number of Monte Carlo samples for ELBO estimation
			:param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
		"""
		super(VAE, self).__init__()
		self.latent_dim = latent_dim
		self.data_dim = data_dim
		self.time_length = time_length

		self.encoder = encoder(latent_dim, encoder_sizes, **kwargs)
		self.decoder = decoder(data_dim, decoder_sizes)
		self.preprocessor = image_preprocessor

		self.beta = beta
		self.K = K
		self.M = M

	def encode(self, x):
		x = tf.Identity(x)  # in case x is not a Tensor already...
		if self.preprocessor is not None:
			x_shape = x.shape.as_list()
			new_shape = [x_shape[0] * x_shape[1]] + list(self.preprocessor.image_shape)
			x_reshaped = tf.reshape(x, new_shape)
			x_preprocessed = self.preprocessor(x_reshaped)
			x = tf.reshape(x_preprocessed, x_shape)
		return self.encoder(x)

	def decode(self, z):
		z = tf.Identity(z)  # in case z is not a Tensor already...
		return self.decoder(z)

	def __call__(self, inputs):
		return self.decode(self.encode(inputs).sample()).sample()

	def generate(self, noise=None, num_samples=1):
		if noise is None:
			noise = tf.random_normal(shape=(num_samples, self.latent_dim))
		return self.decode(noise)
	
	def _get_prior(self):
		if self.prior is None:
			self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim, dtype=tf.float32),
													scale_diag=tf.ones(self.latent_dim, dtype=tf.float32))
		return self.prior

	def compute_nll(self, x, y=None, m_mask=None):
		# Used only for evaluation
		assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
		if y is None: y = x

		z_sample = self.encode(x).sample()
		x_hat_dist = self.decode(z_sample)
		nll = -x_hat_dist.log_prob(y)  # shape=(BS, TL, D)
		nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
		if m_mask is not None:
			m_mask = tf.cast(m_mask, tf.bool)
			nll = tf.where(m_mask, nll, tf.zeros_like(nll))  # !!! inverse mask, set zeros for observed
		return tf.reduce_sum(nll)

	def compute_mse(self, x, y=None, m_mask=None, binary=False):
		# Used only for evaluation
		assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
		if y is None: y = x

		z_mean = self.encode(x).mean()
		x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
		if binary:
			x_hat_mean = tf.round(x_hat_mean)
		mse = tf.math.squared_difference(x_hat_mean, y)
		if m_mask is not None:
			m_mask = tf.cast(m_mask, tf.bool)
			mse = tf.where(m_mask, mse, tf.zeros_like(mse))  # !!! inverse mask, set zeros for observed
		return tf.reduce_sum(mse)

	def _compute_loss(self, x, m_mask=None, return_parts=False):
		assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
		x = tf.Identity(x)  # in case x is not a Tensor already...
		x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)

		if m_mask is not None:
			m_mask = tf.Identity(m_mask)  # in case m_mask is not a Tensor already...
			m_mask = tf.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
			m_mask = tf.cast(m_mask, tf.bool)

		pz = self._get_prior()
		qz_x = self.encode(x)
		z = qz_x.sample()
		px_z = self.decode(z)

		nll = -px_z.log_prob(x)  # shape=(M*K*BS, TL, D)
		nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
		if m_mask is not None:
			nll = tf.where(m_mask, tf.zeros_like(nll), nll)  # if not HI-VAE, m_mask is always zeros
		nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)

		if self.K > 1:
			kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
			kl = tf.where(tf.is_finite(kl), kl, tf.zeros_like(kl))
			kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

			weights = -nll - kl  # shape=(M*K*BS)
			weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)

			elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
			elbo = tf.reduce_mean(elbo)  # scalar
		else:
			# if K==1, compute KL analytically
			kl = self.kl_divergence(qz_x, pz)  # shape=(M*K*BS, TL or d)
			kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
			kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

			elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1
			elbo = tf.reduce_mean(elbo)  # scalar

		if return_parts:
			nll = tf.reduce_mean(nll)  # scalar
			kl = tf.reduce_mean(kl)  # scalar
			return -elbo, nll, kl
		else:
			return -elbo

	def compute_loss(self, x, m_mask=None, return_parts=False):
		del m_mask
		return self._compute_loss(x, return_parts=return_parts)

	def kl_divergence(self, a, b):
		return tfd.kl_divergence(a, b)

	def get_trainable_vars(self):
		self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
						  tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
		return self.trainable_variables


class HI_VAE(VAE):
	""" HI-VAE model, where the reconstruction term in ELBO is summed only over observed components """
	def compute_loss(self, x, m_mask=None, return_parts=False):
		return self._compute_loss(x, m_mask=m_mask, return_parts=return_parts)


class GP_VAE(HI_VAE):
	def __init__(self, *args, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, **kwargs):
		""" Proposed GP-VAE model with Gaussian Process prior
			:param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
			:param sigma: scale parameter for a kernel function
			:param length_scale: length scale parameter for a kernel function
			:param kernel_scales: number of different length scales over latent space dimensions
		"""
		super(GP_VAE, self).__init__(*args, **kwargs)
		self.kernel = kernel
		self.sigma = sigma
		self.length_scale = length_scale
		self.kernel_scales = kernel_scales

		if isinstance(self.encoder, JointEncoder):
			self.encoder.transpose = True

		# Precomputed KL components for efficiency
		self.pz_scale_inv = None
		self.pz_scale_log_abs_determinant = None
		self.prior = None

	def decode(self, z):
		num_dim = len(z.shape)
		assert num_dim > 2
		perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
		return self.decoder(tf.transpose(z, perm=perm))

	def _get_prior(self):
		if self.prior is None:
			# Compute kernel matrices for each latent dimension
			kernel_matrices = []
			for i in range(self.kernel_scales):
				if self.kernel == "rbf":
					kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
				elif self.kernel == "diffusion":
					kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
				elif self.kernel == "matern":
					kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
				elif self.kernel == "cauchy":
					kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

			# Combine kernel matrices for each latent dimension
			tiled_matrices = []
			total = 0
			for i in range(self.kernel_scales):
				if i == self.kernel_scales-1:
					multiplier = self.latent_dim - total
				else:
					multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
					total += multiplier
				tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
			kernel_matrix_tiled = np.concatenate(tiled_matrices)
			assert len(kernel_matrix_tiled) == self.latent_dim

			self.prior = tfd.MultivariateNormalFullCovariance(
				loc=tf.zeros([self.latent_dim, self.time_length], dtype=tf.float32),
				covariance_matrix=kernel_matrix_tiled)
		return self.prior

	def kl_divergence(self, a, b):
		""" Batched KL divergence `KL(a || b)` for multivariate Normals.
			See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
					   /python/distributions/mvn_linear_operator.py
			It's used instead of default KL class in order to exploit precomputed components for efficiency
		"""

		def squared_frobenius_norm(x):
			"""Helper to make KL calculation slightly more readable."""
			return tf.reduce_sum(tf.square(x), axis=[-2, -1])

		def is_diagonal(x):
			"""Helper to identify if `LinearOperator` has only a diagonal component."""
			return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
					isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
					isinstance(x, tf.linalg.LinearOperatorDiag))

		if is_diagonal(a.scale) and is_diagonal(b.scale):
			# Using `stddev` because it handles expansion of Identity cases.
			b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
		else:
			if self.pz_scale_inv is None:
				self.pz_scale_inv = tf.linalg.inv(b.scale.to_dense())
				self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
											 self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

			if self.pz_scale_log_abs_determinant is None:
				self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

			a_shape = a.scale.shape
			if len(b.scale.shape) == 3:
				_b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
			else:
				_b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

			b_inv_a = _b_scale_inv @ a.scale.to_dense()

		# ~10x times faster on CPU then on GPU
		with tf.device('/cpu:0'):
			kl_div = (self.pz_scale_log_abs_determinant - a.scale.log_abs_determinant() +
					  0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
					  squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
					  b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
		return kl_div
'''