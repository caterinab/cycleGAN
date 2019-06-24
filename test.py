import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Attn

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

def toZeroThreshold(x, t=0.1):
    zeros = Tensor(x.shape).fill_(0.0)
    return torch.where(x > t, x, zeros)
	
def test(args):

    transform = transforms.Compose(
        [transforms.Resize((args.crop_height,args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
    AttnA = define_Attn()
    AttnB = define_Attn()
    utils.print_networks([Gab,Gba], ['Gab','Gba'])

    try:
        ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        Gab.load_state_dict(ckpt['Gab'])
        Gba.load_state_dict(ckpt['Gba'])
        AttnA.load_state_dict(ckpt['AttnA'])
        AttnB.load_state_dict(ckpt['AttnB'])
    except Exception as e:
        print(' [*] Checkpoint exception: ' + str(e))


    """ run """
    a_real_test = Variable(iter(a_test_loader).next()[0], requires_grad=True)
    b_real_test = Variable(iter(b_test_loader).next()[0], requires_grad=True)
    a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
            

    Gab.eval()
    Gba.eval()

    with torch.no_grad():
        # A test
        attnMapA = toZeroThreshold(AttnA(a_real_test))      
        fgA = attnMapA * a_real_test                    
        bgA = (1 - attnMapA) * a_real_test                   
        genB = Gba(fgA)        
        b_fake_test = (attnMapA * genB) + bgA
        b_fake_test_copy = b_fake_test.clone()
        attnMapfakeB = toZeroThreshold(AttnB(b_fake_test))   
        fgfakeB = attnMapfakeB * b_fake_test
        bgfakeB = (1 - attnMapfakeB) * b_fake_test
        genA_ = Gab(fgfakeB)
        a_recon_test = (attnMapfakeB * genA_) + bgfakeB         

        # B test
        attnMapB = toZeroThreshold(AttnB(b_real_test))
        fgB = attnMapB * b_real_test
        bgB = (1 - attnMapB) * b_real_test
        genA = Gab(fgB) 
        a_fake_test = (attnMapB * genA) + bgB
        a_fake_test_copy = a_fake_test.clone()
        attnMapfakeA = toZeroThreshold(AttnA(a_fake_test))
        fgfakeA = attnMapfakeA * a_fake_test
        bgfakeA = (1 - attnMapfakeA) * a_fake_test
        genB_ = Gba(fgfakeA)
        b_recon_test = (attnMapfakeA * genB_) + bgfakeA

    pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0
        
    ones = Tensor(a_real_test.shape).fill_(1.0)
    pic2 = (torch.cat([attnMapA*ones, 1 - attnMapA*ones, fgA, bgA, attnMapB*ones, 1 - attnMapB*ones, fgB, bgB], dim=0).data + 1) / 2.0

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    torchvision.utils.save_image(pic, args.results_dir+'/sample.jpg', nrow=3)
    torchvision.utils.save_image(pic2, args.results_dir+'/attn_maps.jpg', nrow=2)