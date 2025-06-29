import numpy as np

class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    Moreover, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        input_grad = module.backward(input, output_grad)
    """
    def __init__ (self):
        self._output = None
        self._input_grad = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        self._output = self._compute_output(input)
        return self._output

    def backward(self, input, output_grad):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self._input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return self._input_grad
    

    def _compute_output(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which will be stored in the `_output` field.

        Example: in case of identity operation:
        
        output = input 
        return output
        """
        raise NotImplementedError
        

    def _compute_input_grad(self, input, output_grad):
        """
        Returns the gradient of the module with respect to its own input. 
        The shape of the returned value is always the same as the shape of `input`.
        
        Example: in case of identity operation:
        input_grad = output_grad
        return input_grad
        """
        
        raise NotImplementedError
    
    def _update_parameters_grad(self, input, output_grad):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zero_grad(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def get_parameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def get_parameters_grad(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"

class BatchNormalization(Module):
    

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0.
        self.moving_variance = 1.
        self.EPS = 1e-3

    def _compute_output(self, input):
        # Your code goes here. ################################################
        
        # output = ...
        N, D = input.shape
        output = np.zeros_like(input)
        if self.training == True:
            batch_mean = np.mean(input,axis = 0)
            batch_variance = np.var(input,axis = 0)
            self.moving_mean = self.moving_mean * self.alpha + batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + batch_variance * (1 - self.alpha)
            for i in range(D):
                output[:,i] = (input[:,i]-batch_mean[i])/np.sqrt(batch_variance[i] + self.EPS)
        else:
            for i in range(D):
                output[:,i] = (input[:,i]-self.moving_mean[i])/np.sqrt(self.moving_variance[i] + self.EPS) 
        return output

    def _compute_input_grad(self, input, output_grad):
        # Your code goes here. ################################################
        N, D = input.shape
        vec_mean = np.mean(input,axis = 0)
        vec_variance = np.var(input,axis = 0)

        # grad_input = np.zeros((D,N))
        # for i in range(D):
        #     batch_mean = vec_mean[i]
        #     batch_variance = vec_variance[i] 
            
        #     # не зависящее от x_i слагаемое
        #     dy1 = N - 1 - (batch_mean**2 + batch_mean/N)/(batch_variance+self.EPS)
        #     dy1 = dy1/(N*np.sqrt(batch_variance+self.EPS))
            
        #     # зависящее от x_i слагаемое
        #     dy2 = input[:,i] - 2*batch_mean-1/N
        #     dy2 = input[:,i]*dy2
        #     dy2 = dy2/(N*np.sqrt(batch_variance+self.EPS))
        #     grad_input[:,i] = dy1 + dy2
        N,D = output_grad.shape
        
        #step7
        
        xmu = (input - vec_mean)
        sqrtvar = np.sqrt(vec_variance+self.EPS)
        
        divar = np.sum(output_grad*xmu, axis=0)
        dxmu1 = output_grad / sqrtvar
        
        #step6
        dsqrtvar = -1. /(sqrtvar**2) * divar
        
        #step5
        dvar = 0.5 * 1. /sqrtvar * dsqrtvar
        
        #step4
        dsq = 1. /N * np.ones((N,D)) * dvar
        
        #step3
        dxmu2 = 2 * xmu * dsq
        
        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        
        #step1
        dx2 = 1. /N * np.ones((N,D)) * dmu
        
        #step0
        dx = dx1 + dx2        
        return dx

    def __repr__(self):
        return "BatchNormalization"
class ChannelwiseScaling(Module):

    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def _compute_output(self, input):
        output = input * self.gamma + self.beta
        return output
        
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.gamma
        return grad_input
    
    def _update_parameters_grad(self, input, output_grad):
        self.gradBeta = np.sum(output_grad, axis=0)
        self.gradGamma = np.sum(output_grad*input, axis=0)
    
    def zero_grad(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def get_parameters(self):
        return [self.gamma, self.beta]
    
    def get_parameters_grad(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = []
        
    def _compute_output(self, input):
        # Your code goes here. ################################################
        
        if self.training == True:
            self.mask = []
            N,D = input.shape
            output = np.zeros_like(input)
            for i in range(N):
                batch_mask = []
                for j in range(D):
                    batch_mask.append(np.random.binomial(1,1-self.p))
                    output[i,j] = input[i,j]*batch_mask[j]
                    output[i,j] = output[i,j]/(1-self.p)
                self.mask.append(batch_mask)
            self.mask = np.array(self.mask)
        else:
            output = input
        return output
    
    def _compute_input_grad(self, input, output_grad):
        # Your code goes here. ################################################
        N,D = input.shape
        grad_input = np.zeros_like(output_grad)
        for i in range(N):
            grad_input[i] = self.mask[i]/(1-self.p)
        return grad_input*output_grad
        
    def __repr__(self):
        return "Dropout"

import scipy.signal as signal

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def _compute_output(self, input):
        n_batch,_,h,w = input.shape
        pad_size = self.kernel_size // 2
        # YOUR CODE ##############################
        # 1. zero-pad the input array
        # 2. compute convolution using scipy.signal.correlate(... , mode='valid')
        # 3. add bias value
        self._output = np.zeros((n_batch,self.out_channels,h,w))
        # print(self._output.shape)
        
        pad_input = np.pad(input,((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)))
        # print(pad_input[0,0])
        for i_batch in range(n_batch):
            for i_out_ch in range(self.out_channels):
                conv_matrix = np.zeros((h,w))
                for i_in_ch in range(self.in_channels):
                    corr = signal.correlate(pad_input[i_batch,i_in_ch],self.W[i_out_ch,i_in_ch],mode='valid')
                    # print(corr)
                    conv_matrix+=corr
                self._output[i_batch,i_out_ch]= conv_matrix + self.b[i_out_ch]
                
        
        return self._output
    
    def _compute_input_grad(self, input, gradOutput):
        n_batch,_,h,w = input.shape
        pad_size = self.kernel_size // 2
        # YOUR CODE ##############################
        # 1. zero-pad the gradOutput
        # 2. compute 'self._input_grad' value using scipy.signal.correlate(... , mode='valid')
        self._input_grad = np.zeros_like(input)
        # print(self._output.shape)
        
        pad_grad = np.pad(gradOutput,((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)))
        inp_grad = np.flip(self.W,axis = (2,3))
        # print(pad_input[0,0])
        for i_batch in range(n_batch):
            for i_in_ch in range(self.in_channels):
                conv_matrix = np.zeros((h,w))
                for i_out_ch in range(self.out_channels):
                    corr = signal.correlate(pad_grad[i_batch,i_out_ch],inp_grad[i_out_ch,i_in_ch],mode='valid')
                    # print(corr)
                    conv_matrix+=corr
                self._input_grad[i_batch,i_in_ch]= conv_matrix
        
        return self._input_grad
    
    def accGradParameters(self, input, gradOutput):
        n_batch = input.shape[0]
        pad_size = self.kernel_size // 2
        # YOUR CODE #############
        # 1. zero-pad the input
        # 2. compute 'self.gradW' using scipy.signal.correlate(... , mode='valid')
        # 3. compute 'self.gradb' - formulas like in Linear of ChannelwiseScaling layers

        pad_input = np.pad(input,((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)))
        self.gradW = np.zeros_like(self.W)
        
        for i_in_ch in range(self.in_channels):
            for i_out_ch in range(self.out_channels):
                conv_matrix = np.zeros((self.kernel_size,self.kernel_size))
                for i_batch in range(n_batch):
                    corr = signal.correlate(pad_input[i_batch,i_in_ch],gradOutput[i_batch,i_out_ch],mode='valid')
                    # print(corr)
                    conv_matrix+=corr
                self.gradW[i_out_ch,i_in_ch]= conv_matrix
                    
        self.gradb = np.sum(gradOutput,axis=(0,2,3))
        pass
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q
