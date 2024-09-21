"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
import torch
import torch.nn.functional as func
from torch import nn

import numpy as np

from tqdm import tqdm

from ..utils.logger import log
from ..utils.timing import Timing, TimingCollector


""" Encoder selection """
def choose_encoder(encoder, nonbinarize, q_in_idlevel, args):
    global encode

    if nonbinarize:
        postfix = " (non binary)"
    else:
        postfix = " (binary)"

    if encoder == 'idlevel':
        # ID Level []
        encode = encode_idlevel
        encode.q = q_in_idlevel
        log.d(">>>>> IDLEVEL encoding" + postfix)
    elif encoder == 'randomproj':
        # random projection []
        encode = encode_random_projection
        log.d(">>>>> Random Projection encoding" + postfix)
    elif encoder == 'nonlinear':
        # nonlinear encoding []
        encode = encode_nonlinear
        log.d(">>>>> Non-linear encoding" + postfix)
    elif encoder == 'hspa':
        # hiearachical sparse encoding
        choose_encoder.args = args
        encode = encode_hierarchical_sparse
        log.d(">>>>> hspa encoding" + postfix)
    elif encoder == 'ae':
        # auto encoding nonlinear
        encode = encode_ae
        log.d(">>>>> auto encoder encoding" + postfix)
    else:
        raise NotImplementedError

    encode.nonbinarize = nonbinarize
    return encode


""" Sign function that only returns -1 or 1 """
def hardsign(x):
    # torch.sign() returns teneray data, i.e., -1, 0, 1, so the valid implementation will be follows
    x_ = torch.ones_like(x)
    x_[x < 0] = -1.0
    return x_


"""
ID-LEVEL encoding
- Imani, Mohsen, Deqian Kong, Abbas Rahimi, and Tajana Rosing. "Voicehd: Hyperdimensional computing for efficient speech recognition." In 2017 IEEE international conference on rebooting computing (ICRC), pp. 1-8. IEEE, 2017.
"""
def encode_idlevel(x, x_test, D):
    batch_size = 64
    F = x.size(1)
    Q = encode.q

    # Build level hypervectors
    base = np.ones(D)
    base[:D//2] = -1.0
    l0 = np.random.permutation(base)
    levels = list() 
    for i in range(Q+1):
        flip = int(int(i/float(Q) * D) / 2)
        li = np.copy(l0)
        li[:flip] = l0[:flip] * -1
        levels.append(li)
    levels = torch.tensor(np.array(levels), dtype=x.dtype, device=x.device)

    # Create base (ID) hypervector 
    bases = []
    for _ in range(F):
        base = np.ones(D)
        base[:D//2] = -1.0
        base = np.random.permutation(base)
        bases.append(base)
    bases = torch.tensor(np.array(bases), dtype=x.dtype, device=x.device)

    def _encode(samples, levels, bases):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            ids = []
            sample = samples[i:i+batch_size]
            level_indices = (sample * Q).long()

            levels_batch = levels[level_indices]
            hv = levels_batch.mul_(bases).sum(dim=1)
            if not encode.nonbinarize:
                H[i:i+batch_size] = hardsign(hv)
            del sample
            del level_indices
            del levels_batch
        return H

    x_h = _encode(x, levels, bases)
    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, levels, bases)

    return x_h, x_test_h


"""
Random projection encoding
- Imani, Mohsen, Yeseong Kim, Sadegh Riazi, John Messerly, Patric Liu, Farinaz Koushanfar, and Tajana Rosing. "A framework for collaborative learning in secure high-dimensional space." In 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), pp. 435-446. IEEE, 2019.
"""
def encode_random_projection(x, x_test, D):
    # Configurations: no impacts on training quality
    batch_size = 512
    F = x.size(1)

    # Create base hypervector
    bases = []
    for _ in range(F):
        base = np.ones(D)
        base[:D//2] = -1.0
        base = np.random.permutation(base)
        bases.append(base)

    bases = torch.tensor(np.array(bases), dtype=x.dtype, device=x.device)
    def _encode(samples, bases):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            torch.matmul(samples[i:i+batch_size], bases, out=H[i:i+batch_size])

        if not encode.nonbinarize:
            H = hardsign(H)
        return H

    x_h = _encode(x, bases)
    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, bases)

    return x_h, x_test_h


"""
Non-linear encoding
- Imani, Mohsen, Saikishan Pampana, Saransh Gupta, Minxuan Zhou, Yeseong Kim, and Tajana Rosing. "Dual: Acceleration of clustering algorithms using digital-based processing in-memory." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 356-371. IEEE, 2020.
"""
def encode_nonlinear(x, x_test, D, y=None):
    # Gaussian sampler configuration
    mu = 0.0
    sigma = 1.0

    # Configurations: no impacts on training quality
    batch_size = 512
    F = x.size(1)

    # Create base hypervector
    bases = torch.empty(D, F, dtype=x.dtype, device=x.device)
    bases = bases.normal_(mu, sigma).T

    def _encode(samples, bases):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            torch.matmul(samples[i:i+batch_size], bases, out=H[i:i+batch_size])
            H[i:i+batch_size].cos_()

            if not encode.nonbinarize:
                H[i:i+batch_size] = hardsign(H[i:i+batch_size])

        return H

    # Encode training and testing dataset
    with Timing("Encode Nonlinear"):
        x_h = _encode(x, bases)
    TimingCollector.g_instance().print()

    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, bases)

    return x_h, x_test_h


"""
Hierarchical propagation step which should be compiled natively for best performance
- It must be located in the outside of the class to get the best performance
  due to the internal management mechanism for the custom operators in pytorch.
"""
@torch.compile(fullgraph=True)
def hierarchical_propagation(H: torch.Tensor, shift: int, threshold: float):
    H_roll = torch.roll(H, shifts=shift, dims=1)
    H.add_(H_roll)
    return torch.where(torch.abs(H) < threshold, 0.0, H)


"""
Hierarchical sparse encoding (DATE'25) - Yeseong Kim, CELL 2024
- Implemented as a torch module
"""
class HierarchicalSparseEncoder(nn.Module):
    def __init__(self, F, D, n_groups, n_depth, **kwargs):
        super(HierarchicalSparseEncoder, self).__init__()

        # Sanity check of the configurations
        assert D % n_groups == 0, "HierarchicalSparseEncoder: D must be divided by n_groups"

        # Main hyperparameters
        self.F = F
        self.D = D
        self.n_groups = n_groups
        self.n_depth = n_depth
        self.randomize_feature_order = True  # Always True for DATE'25
        self.activation_threshold = kwargs.get("activation_threshold", 0.4)
        self.activation_threshold_step = kwargs.get("activation_threshold_step", 0.1)

        device = kwargs.get("device", "cpu")
        self.to(device)

        # Determine leaf-level encoder sizes
        self.n_dims_per_group = D // n_groups
        self.n_features_per_group = F // n_groups
        if F % n_groups != 0:
            self.n_features_per_group += 1
            self.n_features_in_last_group = F % self.n_features_per_group
        else:
            self.n_features_in_last_group = self.n_features_per_group

        # Create gaussian sampler for each group
        mu = 0.0
        sigma = 1.0
        self.bases = torch.empty(
                self.n_groups, self.n_features_per_group, self.n_dims_per_group,
                dtype=torch.float, device=device)
        self.bases = self.bases.normal_(mu, sigma)
        if self.n_features_in_last_group != self.n_features_per_group:
            self.bases[-1, self.n_features_in_last_group:, :] = 0  # add padding

        # Prepare the feature shuffler if randomize_feature_order == True
        if self.randomize_feature_order:
            self.feature_reorder = torch.randperm(F)
            self.feature_reorder.to(device)

    """
    Return the grouped features
    """
    def preprocess_dataset(self, x):
        if self.randomize_feature_order:
            x = x[:, self.feature_reorder]
        pad_size = self.n_features_per_group - self.n_features_in_last_group
        x_padded = func.pad(x, (0, pad_size))
        x_grouped = x_padded.reshape(x.size(0), self.n_groups, self.n_features_per_group)
        x_grouped = x_grouped.contiguous()  #  better align
        return x_grouped


    """
    Encode proccess
    - x: preprocessed dataset, i.e., (BATCH, N_GROUPS, GROUP_FEATURES)
    """
    def forward(self, x):
        # Step 1: Nonlinear encoding for all groups - leaf-level
        # - bmm for (G, B, F') and (G, F', D')
        #   where B is batch size, G is # groups, and
        #   F' is the # features/group,
        #   resulting in (G, B, D')
        H = torch.bmm(x.transpose(0, 1), self.bases)  
        H = H.transpose(0, 1)  # make (B, G, D)
        H = H.contiguous()  # Make it computation-friedly for torch.roll
        #H = sparsity_inducing_activation(H)

        # Step 2. Hierarchical propagation
        for d in range(self.n_depth):
            thre = self.activation_threshold + d * self.activation_threshold_step
            H = hierarchical_propagation(H, (2**d), thre)

            # hierarchical_propagation equally performs:
            #h_roll = torch.roll(H, shifts=(2 ** d), dims=1)
            #H += h_roll
            #H = torch.where(torch.abs(H) < thre, 0.0, H)

        # We can't apply the cos similarity here, as our thresholding function
        # will make many values around 0, but the cos function makes them to 1 again
        #H = torch.cos(H)
        
        if not encode.nonbinarize:
            H = torch.sign(H)

        return H.reshape(H.size(0), self.D)


"""
Hierarchical sparse encode wrapper
"""
def encode_hierarchical_sparse(x, x_test, D, y=None):
    # Rename command argument variable (given in choose_encoder)
    args = choose_encoder.args

    # Configurations
    batch_size = 512  # no impacts on training quality
    F = x.size(1)

    # Prepare encoder class instance
    encoder = HierarchicalSparseEncoder(
            F, D, args.n_groups_hspa, args.n_depth_hspa,
            activation_threshold=args.activation_threshold,
            activation_threshold_step=args.activation_threshold_step,
            dtype=x.dtype, device=x.device)
    encoder.eval()  # disable backprop

    x_grouped = encoder.preprocess_dataset(x)
    if x_test is not None:
        x_test_grouped = encoder.preprocess_dataset(x_test)

    # Main encoding function
    def _encode(samples, encoder):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            H_enc = encoder(samples[i:i+batch_size])
            H[i:i+batch_size] = H_enc
        return H

    # TODO: Better ahead-of-time compile to measure the time?
    # We use a custom operation for hierarchical propagationm step, which
    # needs to be compiled. For accurate timing measurement w/o such 
    # compile time, the compile should happen ahead of time; 
    # but currently I'm not sure how to enforce the compiling process.
    # So, I instead run the entire encoding process once for warm-up process
    # so that it compiles the required kernels, and measure the time later.
    x_h = _encode(x_grouped, encoder)

    # Encode training and testing dataset
    with Timing("Encode HSPA"):
        x_h = _encode(x_grouped, encoder)
    TimingCollector.g_instance().print()

    print("Sparsity : {}".format(torch.sum(x_h==0) / torch.numel(x_h)))

    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test_grouped, encoder)

    return x_h, x_test_h


def encode_ae(x, x_test, D, y=None):
    # Gaussian sampler configuration
    mu = 0.0
    sigma = 1.0

    d_prime = 100

    # Configurations: no impacts on training quality
    batch_size = 512
    F = x.size(1)

    # Create base hypervector
    bases1 = torch.empty(F, d_prime, dtype=x.dtype, device=x.device)
    bases1 = bases1.normal_(mu, sigma)

    bases2 = torch.empty(d_prime, D, dtype=x.dtype, device=x.device)
    bases2 = bases2.normal_(mu, sigma)

    def _encode(samples, bases1, bases2):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            H_prime = torch.matmul(samples[i:i+batch_size], bases1)
            #H_prime = func.normalize(H_prime, p=2)
            torch.matmul(H_prime, bases2, out=H[i:i+batch_size])
            #H[i:i+batch_size] = H_prime
            #H[i:i+batch_size] *= np.sqrt(np.pi/4) #d_prime * F)
            H[i:i+batch_size].cos_()

            #torch.set_printoptions(precision=2)
            #torch.set_printoptions(edgeitems=10)
            #print(H[i:i+batch_size][0])
            #print(torch.sum(H[i:i+batch_size]>0))
            #print(H[i:i+batch_size].size())
            ##print(bases)
            #exit()

            if not encode.nonbinarize:
                H[i:i+batch_size] = hardsign(H[i:i+batch_size])
        return H

    # Encode training and testing dataset
    with Timing("Encode AE"):
        x_h = _encode(x, bases1, bases2)
    TimingCollector.g_instance().print()

    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, bases1, bases2)

    return x_h, x_test_h


"""
!!!!!!!! The following class are not used at this moment !!!!!!!!!
- I kept this as to keep the note and development process,
  which could be valuable to revisit,
  But for the DATE 25's Sparsity version, 
  use the HierarchicalSparseEncoder class instead

Hierarchical propagation encoding (more comprehensive) - Yeseong Kim, CELL 2024
- Implemented as a torch module
"""
#class HierarchicalPropagationEncoderTest(nn.Module):
#    def __init__(self, F, D, n_groups, randomize_feature_order=True, **kwargs):
#        super(HierarchicalPropagationEncoderTest, self).__init__()
#
#        # Sanity check of the configurations
#        assert D % n_groups == 0, "D must be divided by n_groups"
#
#        # Main hyperparameters
#        self.F = F
#        self.D = D
#        self.n_groups = n_groups
#        self.randomize_feature_order = randomize_feature_order
#
#        self.receptive_field_size = kwargs.get("receptive_field_size", 1)
#        self.propagation_decay = kwargs.get("propagation_decay", 0.7)
#        self.activation_threshold = kwargs.get("activation_threshold", 0.5)
#        self.use_soft_threshold = kwargs.get("use_soft_threshold", True)
#
#        if self.receptive_field_size != 1:
#            # Note: theoretically it is possible, but we currently implemented
#            # the field size 1 for the simplicity and computation efficiency
#            # (I also think there is no practical gain in accuracy even if
#            # it is more than 1. But, probably convolution-like data structure
#            # would require the support of these cases.)
#            raise NotImplemented  
#
#
#        device = kwargs.get("device", "cpu")  # TODO: check if it works or needed
#        self.to(device)
#
#        # Determine leaf-level encoder sizes
#        self.n_dims_per_group = D // n_groups
#        self.n_features_per_group = F // n_groups
#        if F % n_groups != 0:
#            self.n_features_per_group += 1
#            self.n_features_in_last_group = F % self.n_features_per_group
#        else:
#            self.n_features_in_last_group = self.n_features_per_group
#
#        # Create gaussian sampler for each group
#        # TODO: CSR-aware aligned shuffling for base matrix?
#        mu = 0.0
#        sigma = 1.0
#        self.bases = torch.empty(
#                self.n_groups, self.n_features_per_group, self.n_dims_per_group,
#                dtype=torch.float, device=device)
#        self.bases = self.bases.normal_(mu, sigma)
#        if self.n_features_in_last_group != self.n_features_per_group:
#            self.bases[-1, self.n_features_in_last_group:, :] = 0  # add padding
#
#        # Prepare the feature shuffler if randomize_feature_order == True
#        if self.randomize_feature_order:
#            self.feature_reorder = torch.randperm(F)
#            self.feature_reorder.to(device)
#
#
#    """
#    Return the grouped features
#    """
#    def preprocess_dataset(self, x):
#        if self.randomize_feature_order:
#            x = x[:, self.feature_reorder]
#        pad_size = self.n_features_per_group - self.n_features_in_last_group
#        x_padded = func.pad(x, (0, pad_size))
#        x_grouped = x_padded.reshape(x.size(0), self.n_groups, self.n_features_per_group)
#        return x_grouped
#
#
#    """
#    Encode proccess
#    - x: preprocessed dataset, i.e., (BATCH, N_GROUPS, GROUP_FEATURES)
#    """
#    def forward(self, x):
#        # internal function to apply sparsity-inducing activation (thresholding)
#        def sparsity_inducing_activation(H):
#            if self.use_soft_threshold:
#                H = torch.sign(H) * torch.relu(torch.abs(H) - self.activation_threshold) / (1 - self.activation_threshold)
#            else:
#                H[torch.abs(H) < self.activation_threshold] = 0
#            H = torch.clamp(H, -1, 1)  # TODO: Check if works
#            return H
#
#        # Step 1: Nonlinear encoding for all groups - leaf-level
#        # - bmm for (G, B, F') and (G, F', D)
#        #   where B is batch size, G is # groups, and F' is the # features/group
#        #   resulting in (G, B, D)
#        H = torch.bmm(x.transpose(0, 1), self.bases)  
#        H = H.transpose(0, 1)  # make (B, G, D)
#
#        #torch.set_printoptions(precision=2)
#        #torch.set_printoptions(edgeitems=10)
#        #print(H)
#        #print(torch.sum(H>0))
#        #print(H.size())
#        #exit()
#
#        #H = torch.cos(H)
#        #H = sparsity_inducing_activation(H)
#        #return torch.sign(H.reshape(H.size(0), self.D))
#
#        # Step 2. Hierarchical propagation for receptive field increase
#        # - Future Idea: Instead of weighting the receptive field,
#        #                we may apply attention and learn them
#        # - Note: Current implementation for self.receptive_field_size == 1
#        propagation_depths = self.n_groups - 1
#
#
#        H_r = torch.roll(H, shifts=1, dims=1)
#        for d in range(5): #propagation_depths/2):
#            #H_l = torch.roll(H, shifts=-1, dims=1)
#
#            # Note: If you want to apply backprop or other learning methoa,
#            # the self.propagation_decay should be a form of tensors so that 
#            # it learns the different weight factors for each propagation
#            H += self.propagation_decay * H_r
#            #H += self.propagation_decay * H_l
#            #H = sparsity_inducing_activation(H)
#            H_r = torch.roll(H, shifts=1, dims=1)
#            H = sparsity_inducing_activation(H)
#
#        #H = torch.cos(H)
#        #H = torch.sign(H)
#        #H = sparsity_inducing_activation(H)
#
#        #torch.set_printoptions(precision=2)
#        #torch.set_printoptions(edgeitems=10)
#        #print(H)
#        print(H.size())
#        print(torch.sum(H>0))
#        print(torch.sum(H==0))
#        print(torch.numel(H))
#        print(torch.sum(H==0) / torch.numel(H))
#        #exit()
#
#        return H.reshape(H.size(0), self.D)
