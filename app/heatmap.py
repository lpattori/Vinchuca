from fastai.vision import *
from fastai.callbacks.hooks import *
from torchvision import transforms

def scale_down (img_pil, pixel: int):
    img_w, img_h = img_pil.size
    if min(img_w, img_h) > pixel:
        ratio = pixel / min(img_w, img_h)
        img_pil = img_pil.resize((int(img_w * ratio), int(img_h * ratio)),
                                 resample=PIL.Image.BILINEAR).convert('RGB')
    return Image(pil2tensor(img_pil.convert("RGB"), np.float32).div_(255))



def hooked_backward(m , img: Image.data, cat: int):
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(img)
            preds[0, cat].backward()
    return hook_a, hook_g


def heatmap(learner: Learner, img: Image, pred_idx: int, first: bool) -> PIL.Image.Image:
    m = learner.model.eval();
    hook_a, hook_g = hooked_backward(m, img.data[None], pred_idx)

    if first:
        acts = hook_a.stored[0].cpu()
        data = acts.mean(0).numpy()
    else:
        acts = hook_a.stored[0].cpu().numpy()
        target_grad = hook_g.stored[0][0].cpu().numpy()
        mean_grad = target_grad.mean(1).mean(1)
        #     data = (acts * mean_grad[...,None,None]).mean(0)
        data = (acts * mean_grad[..., None, None]).sum(0)

    max_pixel = data.max()
    min_pixel = data.min()

    data = (data- min_pixel) / (max_pixel - min_pixel)
    # Get the color map by name:
    cm = plt.get_cmap('magma')
    # Apply the colormap like a function to any array:
    colored_image = cm(data)
    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it:
    colored_PIL = PIL.Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
    colored_PIL = colored_PIL.resize((img.size[1],img.size[0]), resample=PIL.Image.BILINEAR)
    img_PIL = img.data
    img_PIL = transforms.ToPILImage()(img_PIL)
    blend = PIL.Image.blend(img_PIL, colored_PIL, 0.6)
    return blend

