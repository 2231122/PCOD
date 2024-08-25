import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps
import numpy as np

criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
def flood(label,P_fg,x,y,h,count):
    #label:only 1 and 0
    label=1-label#翻转前景和其他
    det_p=label*P_fg[:,1:2,:,:]

    book=label*0
    book1=label*0+1
    N,C,W,H=label.shape

    #320 ?
    m_x=H
    m_y=W
    #x and y is [16,1]
    l_up_x = np.zeros(N).astype(int)
    l_up_y = np.zeros(N).astype(int)
    for i in range(N):
        l_up_x[i] = x[i] - h
        l_up_y[i] = y[i] - h
        if l_up_x[i]<0:
            l_up_x[i]=0
        if l_up_y[i]<0:
            l_up_y[i]=0
        #next ???
        if l_up_x[i]+2*h>m_x-1:
            l_up_x[i]=m_x-1-2*h
        if l_up_y[i]+2*h>m_y-1:
            l_up_y[i]=m_y-1-2*h
        #print=(h)
        #y's suoyin is at front !

        # book[i,0,l_up_x[i],l_up_y[i]:l_up_y[i]+2*h]=1
        # book[i,0,l_up_x[i]+2*h,l_up_y[i]:l_up_y[i]+2*h]=1
        # book[i,0,l_up_x[i]:l_up_x[i]+2*h,l_up_y[i]]=1
        # book[i,0,l_up_x[i]:l_up_x[i]+2*h,l_up_y[i]+2*h]=1

        book[i, 0, l_up_y[i], l_up_x[i]:l_up_x[i] + 2 * h] = 1
        book[i, 0, l_up_y[i] + 2 * h, l_up_x[i]:l_up_x[i] + 2 * h] = 1
        book[i, 0, l_up_y[i]:l_up_y[i] + 2 * h, l_up_x[i]] = 1
        book[i, 0, l_up_y[i]:l_up_y[i] + 2 * h, l_up_x[i] + 2 * h] = 1

    soft_label=book*det_p
    book1=book1-soft_label
    count=count+np.sum(soft_label)

    return soft_label,book1,count
def softlabel(pred,label,epoch):
    #thr=0.8
    #label情况：1:前景，0:背景，255不知道
    # thr:Conservative or not !!!
    P_soft_fg=(pred.cpu().detach().numpy()>0.95).astype(int)#the thr is a hy paramater !!!
    fg=(label.cpu().numpy()==1).astype(int)
    N,_,W,H=label.size()
    # P_soft_bg=pred<0.1
    center_y=np.zeros(N)
    center_x=np.zeros(N)
    for i in range(N):
        b_f=fg[i,:,:,:]
        P = b_f.argmax()
        center_y[i] = P // H
        center_x[i] = P % H



    #center_x,center_y=center_x+5,center_y+5
    #N is mast be change to correct the result !!!
    N=300 #f(epoch,size?)
    count=0
    soft_label=label
    #or cu_label is 11x11!
    for l in range(6,15):
        soft_label_l,mut_p,count=flood(fg,P_soft_fg,center_x,center_y,l,count)
        if count<N:
            #soft_label has 255!!! 0,1,255;and all most of is 255 !!!

            soft_label=soft_label.cpu()*mut_p+soft_label_l
        else:
            break
    return soft_label
def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def unsymmetric_grad(x, y, calc, w1, w2):
    '''
    x: strong feature
    y: weak feature'''
    return calc(x, y.detach())*w1 + calc(x.detach(), y)*w2

# targeted at boundary: only p/n coexsits.
# learn features that focus on boundary prediction
# use feature vectors to guide pixel prediction 
# covariance to encourage the feature difference in the most decisive ones
# def feature_loss(feature_map, pred, kr=4, norm=False, crtl_loss=True, w_ftp=0, topk=16, step_ratio=2):
#     '''
#     pred: n, 1, h, w'''
#     # normalize feature map (but how?)
#     if norm:
#         fmap = feature_map / feature_map.std(dim=(-1,-2), keepdim=True).mean(dim=1, keepdim=True)
#     else: fmap=feature_map
#     # print(fmap.max(), fmap.min(), fmap.std(dim=(-1,-2)).max())
#
#     n, c, h, w =fmap.shape
#     # get local feature map
#     ks = 2*kr
#     assert h%ks==0 and w%ks==0
#     # print('ks', ks)
#     uf = lambda x: F.unfold(x, ks, padding = 0, stride=ks//step_ratio).permute(0,2,1).reshape(-1, x.shape[1], ks*ks) # N * no.blk, 64, 8*8
#     fcmap = uf(fmap)
#     fcpred = uf(pred) # N', 1, 10*10
#     # # get fg/bg confident coexisting block
#     cfd_thres = .8
#     exst = lambda x: (x>cfd_thres).sum(2, keepdim=True) > 0.3*ks*ks
#     coexists = (exst(fcpred) & exst(1-fcpred))
#     coexists = coexists[:, 0, 0] # N', 1, 1
#
#     fcmap = fcmap[coexists]
#     fcpred = fcpred[coexists]
#     # print(fcmap.shape, fcpred.shape)
#     if not len(fcmap):
#         return 0, 0
#     # minus mean
#     mfcmap = fcmap - fcmap.mean(2, keepdim=True)
#     mfcpred = fcpred - fcpred.mean(2, keepdim=True)
#     # get most relevance in confident area bout saliency
#     cov = mfcmap.matmul(mfcpred.permute(0, 2, 1)) # N', 64, 1
#     sgnf_id = cov.abs().topk(topk, dim=1)[1].expand(-1,-1,ks*ks) # n', topk, 10*10
#     sg_fcmap = fcmap.gather(dim=1, index=sgnf_id) # n', topk, 10*10
#     # different potential calculation
#     crf_k = lambda x: (-(x[:, :, None]-x[:, :, :, None])**2 * 0.5).sum(1, keepdim=True).exp() # n', 1, 100, 100
#     pred_grvt = lambda x,y: (1-x)*y + x*(1-y) # (x-y).abs() # x*y + (1-x)*(1-y) - x*(1-y) - (1-x)*y
#     ft_grvt = lambda x: 1-crf_k(x)
#     # position
#     xy = torch.stack(torch.meshgrid(torch.arange(ks, device=pred.device), torch.arange(ks, device=pred.device))) / 6
#     xy = (xy).reshape(1,2, ks*ks).expand(len(sg_fcmap),-1,-1) # 1, 1, 100
#     ffxy = crf_k(xy)
#     if crtl_loss:
#         # train the feature map without pred grad
#         # L2 norm loss
#         pmap = fcpred.detach()
#         pmap = 0.5 - pred_grvt(pmap.unsqueeze(2), pmap.unsqueeze(-1)) # n', 1, 100, 100
#         fpmap = ft_grvt(sg_fcmap) * ffxy
#         ice = (pmap*fpmap).mean()
#         # reversely, train the pred map
#         # calculate CRF with confident point
#         fffm = crf_k(sg_fcmap.detach())
#         kernel = fffm*ffxy # n', 1, 10*10, 10*10
#     else:
#         ice = 0
#         fffm = crf_k(sg_fcmap)
#         kernel = fffm*ffxy # n', 1, 10*10, 10*10
#         kernel[torch.eye(ks*ks, device=pred.device, dtype=bool).expand_as(kernel)] = 0
#
#     pp = pred_grvt(fcpred[:,:,None], fcpred.unsqueeze(-1)) # n', 1, 100, 100
#     if w_ftp==0:
#         crf = (kernel * pp).mean()
#     elif w_ftp==1:
#         crf = (kernel.detach() * pp).mean() * (1+w_ftp)
#     else:
#         crf = unsymmetric_grad(kernel, pp, lambda x,y:(x*y).mean(), 1-w_ftp, 1+w_ftp)
#     return crf, ice

def train_loss(image, mask, net, ctx, ft_dct, w_ft=.1, ft_st = 2, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):

    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']
    ### feature loss
    # fm = []
    # def hook(m, i, o):
    #     if not ft_head:
    #         fm.extend(get_featuremap(m, i[0]))
    #     else:
    #         fm.append(net.feature_head[0](i[0]))
    # hh = net.head[0].register_forward_hook(hook)#init
    #
    # hh = net.module.head[0].register_forward_hook(hook)

    ######  saliency structure consistency loss  ######
    do_moretrsf = np.random.uniform()<mtrsf_prob
    if do_moretrsf:
        pre_transform = get_transform(ops)
        image_tr = pre_transform(image)
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
    sc_fct = 0.6 if large_scale else 0.3
    #image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)
    #out2, loss_c , out3, out4, out5, out6, hook0 ,fg ,bg = net(image, )
    out2, loss_c, out3, out4, out5, out6, hook0 = net(image, )

    #get_featuremap()
    # out2_org = out2

    #hh.remove()
    #out2_s, _, out3_s, out4_s, out5_s, out6_s,_ ,fg_s,bg_s= net(image_scale, )
    #out2_s, _, out3_s, out4_s, out5_s, out6_s, _ = net(image_scale, )
    ### Calc intra_consisten loss (l2 norm) / entorpy
    loss_intra = []
    #me_st too large >50 epoch
    if epoch>=me_st:
        def entrp(t):
            etp = -(F.softmax (t, dim=1) * F.log_softmax (t, dim=1)).sum(dim=1)
            msk = (etp<0.5)
            return (etp*msk).sum() / (msk.sum() or 1)
        me = lambda x: entrp(torch.cat((x*0, x), 1)) # orig: 1-x, x
        if not me_all:
            e = me(out2)
            loss_intra.append(e * get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st))
            loss_intra = loss_intra + [0,0,0,0]
            sw.add_scalar('intra entropy', e.item(), global_step)
        elif me_all:
            ga = get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st)
            for i in [out2, out3, out4, out5, out6]:
                loss_intra.append(me(i)*ga)
            sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0 for _ in range(5)])

    # def out_proc(out2, out3, out4, out5, out6,fg,bg):
    #     a = [out2, out3, out4, out5, out6,fg,bg]

    def out_proc(out2, out3, out4, out5, out6):
        a = [out2, out3, out4, out5, out6]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a
    #sigmoid
    out2, out3, out4, out5, out6 = out_proc(out2, out3, out4, out5, out6) #init

    #out2, out3, out4, out5, out6,fg,bg = out_proc(out2, out3, out4, out5, out6,fg,bg)
    # the size of out_s is be transformered
    #out2_s, out3_s, out4_s, out5_s, out6_s,fg_s,bg_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s,fg_s,bg_s)
    #out2_s, out3_s, out4_s, out5_s, out6_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s)
    # if not do_moretrsf:
    #     out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
    #     out2_s = out2_s[:, 1:2]
    #     # out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    # else:
    #     out2_ss = pre_transform(out2)
    #     out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
    #     out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    # wen ding xing f
    #loss_ssc = (SaliencyStructureConsistency(out2_s, out2_scale.detach(), 0.85) * (w_l2g + 1) + SaliencyStructureConsistency(out2_s.detach(), out2_scale, 0.85) * (1 - w_l2g)) if sl else 0
    #loss_ssc=loss_ssc*0

    ######   label for partial cross-entropy loss  ######
    gt = mask.squeeze(1).long()
    #mask has
    bg_label = gt.clone()
    fg_label = gt.clone()
    #1=foreground 2=background 0=unknown_pixels,

    # if epoch>10:
    #     gt = softlabel(out2, mask, epoch)
    #     gt = gt.squeeze(1).long()
    #0=bg ,1=fg ,255=unknown
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255

    #gt_c=softlabel()
    #bg_label[gt_c!=0]=255
    #fg_label[gt_c==0]=255
    #m=0.9
    #loss=(1-m)*(criterion(out2, fg_label) + criterion(out2, bg_label))

    #feature losses ; but it doesn't be used in the projection.
    ## feature loss
    # if epoch>=ft_st:
    #     wl = get_current_consistency_weight(epoch-ft_st, w_ft, t_epo-ft_st)
    #     # ft_map, bs = fm
    #
    #     ft_map =hook0#(fm[0])#7
    #     #print(ft_map.shape)# [8,64,80,80]
    #     #print(hook0.shape) #[16,64,80,80]
    #     #out2=14
    #     pred_s = out2[:, 1:2].clone()#14
    #     pred_s[:,0][gt!=255] = gt[gt!=255].float()
    #     pred_s = F.interpolate(pred_s, scale_factor = ft_fct, mode='bilinear', align_corners=False)
    #     # adjust size
    #
    #     ft_map = F.interpolate(ft_map, out2.shape[-2:], mode='bilinear', align_corners=False)
    #     ft_map = F.interpolate(ft_map, pred_s.shape[-2:], mode='bilinear', align_corners=False)
    #
    #
    #
    #     fl, crtl = feature_loss(ft_map, pred_s, **ft_dct)
    #     # print('here', loss_ssc, crtl, fl, wl)
    #     sw.add_scalar('ft_loss', fl.item() if isinstance(fl, torch.torch.Tensor) else fl, global_step=global_step)
    #     sw.add_scalar('fthead_loss', crtl.item() if isinstance(crtl, torch.torch.Tensor) else crtl, global_step=global_step)
    #     loss_ssc = loss_ssc + crtl + fl * wl


    ######   local saliency coherence loss (scale to realize large batchsize)  ######
    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
    sample = {'rgb': image_}
    # print('sample :', image_.max(), image_.min(), image_.std())
    #out2_ = F.interpolate(out2[:, 0:1], scale_factor=0.25, mode='bilinear', align_corners=True)
    out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss2 = (criterion(out2, fg_label) + criterion(out2, bg_label)) + l * loss2_lsc + loss_intra[0]
    #loss2 = loss_ssc + (criterion(out2, fg_label) + criterion(out2, bg_label)) + l * loss2_lsc + loss_intra[0] ## dominant loss
    #loss_Mts = criterion(fg,fg_label)+criterion(bg,bg_label)
    "all auxiliary losses need to be reconstructed"
    ######  auxiliary losses  ######
    # out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    # loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    # loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc + loss_intra[1]
    # out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    # loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    # loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc + loss_intra[2]
    # out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    # loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    # loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * loss5_lsc + loss_intra[3]
    #
    # out6_ = F.interpolate(out6[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    # loss6_lsc = loss_lsc(out6_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    # loss6 = criterion(out6, fg_label) + criterion(out6, bg_label) + l * loss6_lsc + loss_intra[4]

    return loss2, loss_c, loss2*0.0, loss2*0.0, loss2*0.0