import itertools
import functools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis, set_grad, define_Attn
from torch.optim import lr_scheduler

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

'''
Class for CycleGAN with train() as a member function

'''

def toZeroThreshold(x, t=0.1):
    zeros = Tensor(x.shape).fill_(0.0)
    return torch.where(x > t, x, zeros)

class cycleGAN(object):
    def __init__(self,args):

        # Define the network 
        #####################################################
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        
		# Attention Modules
        self.AttnA = define_Attn()
        self.AttnB = define_Attn()

        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        
        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))

        # Optimizers - attn
        self.optAttn = torch.optim.Adam(itertools.chain(self.AttnA.parameters(), self.AttnB.parameters()),lr=args.LRattn)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            self.AttnA.load_state_dict(ckpt['AttnA'])
            self.AttnB.load_state_dict(ckpt['AttnB'])
            self.optAttn.load_state_dict(ckpt['optAttn'])
        except Exception as e:
            print(' [*] Checkpoint exception: ' + str(e))
            self.start_epoch = 0

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.a_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optAttn, milestones=[30], gamma=0.1, last_epoch=self.start_epoch-1)
        
    def train(self,args):
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height,args.load_width)),
             transforms.RandomCrop((args.crop_height,args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                # step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()
                self.optAttn.zero_grad()

                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])
                a_real, b_real = utils.cuda([a_real, b_real])
				
				# NB: Gab and Gba may have inverted names (Gab generates a from b, not the other way around) 
				
                # A
                attnMapA = toZeroThreshold(self.AttnA(a_real))        # compute attention on real A
                fgA = attnMapA * a_real                            # foreground
                bgA = (1 - attnMapA) * a_real                    # background
                genB = self.Gba(fgA)                            # generate fake B using foreground only
                fakeB = (attnMapA * genB) + bgA                    # add background to generated fake B
                fakeBcopy = fakeB.clone()
                attnMapfakeB = toZeroThreshold(self.AttnB(fakeB))    # (repeat from first step to reconstruct image)
                fgfakeB = attnMapfakeB * fakeB
                bgfakeB = (1 - attnMapfakeB) * fakeB
                genA_ = self.Gab(fgfakeB)
                A_ = (attnMapfakeB * genA_) + bgfakeB            # A_ is A reconstructed
                
                # B
                attnMapB = toZeroThreshold(self.AttnB(b_real))
                fgB = attnMapB * b_real
                bgB = (1 - attnMapB) * b_real
                genA = self.Gab(fgB) 
                fakeA = (attnMapB * genA) + bgB
                fakeAcopy = fakeA.clone()
                attnMapfakeA = toZeroThreshold(self.AttnA(fakeA))
                fgfakeA = attnMapfakeA * fakeA
                bgfakeA = (1 - attnMapfakeA) * fakeA
                genB_ = self.Gba(fgfakeA)
                B_ = (attnMapfakeA * genB_) + bgfakeA

                # Forward pass through generators
                ##################################################
                '''
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)
                '''
                # Identity losses
                ###################################################
                #a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                #b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef
                
                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(fakeA)	# generated A
                a_fake_dis_ = self.Da(A_)	# reconstructed A
                b_fake_dis = self.Db(fakeB)
                b_fake_dis_ = self.Db(B_)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label) + self.MSE(a_fake_dis_, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label) + self.MSE(b_fake_dis_, real_label)

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_real, A_) * args.lamda
                b_cycle_loss = self.L1(b_real, B_) * args.lamda

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss #+ a_idt_loss + b_idt_loss

                # Update generators
                ###################################################
                gen_loss.backward(retain_graph=True)
                self.g_optimizer.step()
                self.optAttn.step()

                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()
                
                # Sample from history of generated images
                #################################################
                fakeA = Variable(torch.Tensor(a_fake_sample([fakeA.cpu().data.numpy()])[0]))
                fakeB = Variable(torch.Tensor(b_fake_sample([fakeB.cpu().data.numpy()])[0]))
                fakeA, fakeB = utils.cuda([fakeA, fakeB])

                # Forward pass through discriminators
                ################################################# 
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(fakeA)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(fakeB)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                a_dis_fake_loss_ = self.MSE(a_fake_dis_, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)
                b_dis_fake_loss_ = self.MSE(b_fake_dis_, fake_label)                

                # Total discriminators losses
                a_dis_loss = a_dis_fake_loss + a_dis_fake_loss_ + 2*a_dis_real_loss
                b_dis_loss = b_dis_fake_loss + b_dis_fake_loss_ + 2*b_dis_real_loss
                #a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                #b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5
                
                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                            (epoch, i + 1, min(len(a_loader), len(b_loader)),
                                                            gen_loss,a_dis_loss+b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict(),
                                   'AttnA': self.AttnA.state_dict(),
                                   'AttnB': self.AttnB.state_dict(),
                                   'optAttn': self.optAttn.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            self.a_lr_scheduler.step()



