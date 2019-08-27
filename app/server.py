import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = ('https://onedrive.live.com/download?'
                   'cid=27B3CFFF6EE897C2&resid=27B3CFFF6EE897C2%2119427&authkey=AOCpzm2XOxwezP0')
export_file_name = 'vinchuca.pkl'
export_root_name = 'vinchuca'

export_file_url_ant = ('https://onedrive.live.com/download?'
                       'cid=27B3CFFF6EE897C2&resid=27B3CFFF6EE897C2%2119429&authkey=AD_G8RzdMUV9l_w')
export_file_name_ant = 'anterior.pkl'
export_root_name_ant = 'anterior'

classes = ['No vinchuca','Vinchuca']
path = Path(__file__).parent
path_model = path / 'models'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path_model / export_file_name)
    await download_file(export_file_url_ant, path_model / export_file_name_ant)
    try:
        learn = load_learner(path_model, export_file_name)
        learn_ant = load_learner(path_model, export_file_name_ant)
        return learn, learn_ant
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn, learn_ant = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img_pil = PIL.Image.open(img_bytes)
    if max(img_pil) > 322 :
        img_pil = img_pil.resize(322, resample=PIL.Image.BILINEAR).convert('RGB')
    img = Image(pil2tensor(img_pil.convert("RGB"), np.float32).div_(255))
    pred_clase, pred_idx, salida = learn.predict(img)
    pred_clase_ant, pred_idx_ant, salida_ant = learn_ant.predict(img)
    prediccion = "%s: %.2f %% \n" % (pred_clase, salida[pred_idx] * 100)
    predi_ant =  "%s: %.2f %% \n" % (pred_clase_ant, salida_ant[pred_idx_ant] * 100)
    return JSONResponse({'result': prediccion, 'anterior': predi_ant})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        puerto=int(sys.argv[2])
        uvicorn.run(app=app, host='0.0.0.0', port=puerto, log_level="info")
