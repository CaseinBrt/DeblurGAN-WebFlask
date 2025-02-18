import torch
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.opt = opt
        self.device = torch.device('cuda' if (torch.cuda.is_available() and len(opt.gpu_ids) > 0) else 'cpu')
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.gpu_ids, False,
                                      opt.learn_residual)
        self.netG = self.netG.to(self.device)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        self.input_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.fake_B = self.netG.forward(self.real_A)
            # Ensure fake_B has the same size as real_A
            if self.fake_B.size() != self.real_A.size():
                self.fake_B = nn.functional.interpolate(self.fake_B, size=self.real_A.size()[2:], mode='bilinear', align_corners=False)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
