def _jpeg_compression(im):
    assert torch.is_tensor(im)
    im = ToPILImage()(im)
    savepath = BytesIO()
    im.save(savepath, 'JPEG', quality=75)
    im = Image.open(savepath)
    im = ToTensor()(im)
    return im