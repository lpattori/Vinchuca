import aiohttp
import asyncio
import uvicorn
import csv
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

class Aprendizaje():
    "Estructura para guardar learner nombre y clase asociadas"
    def __init__(self, learner, nombre, descripcion):
        self.learner = learner
        self.nombre = nombre
        self.descripcion = descripcion


csv.register_dialect('no_quotes', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)
csv_file_url = ('https://onedrive.live.com/download?'
                'cid=27B3CFFF6EE897C2&resid=27B3CFFF6EE897C2%2122311&authkey=AFQ1XNRERUjj8eI')
csv_file_name = 'parametros.csv'
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
    await download_file(csv_file_url, path_model / csv_file_name)
    with open(path_model / csv_file_name, 'r') as descripcion:
        reader = csv.reader(descripcion, dialect='no_quotes')
        lista_redes = list(reader)
    for nombre_learner, descripcion_learner, onedrive_url in lista_redes:
        await download_file(onedrive_url, path_model / nombre_learner)
        try:
            lista_learn.append(Aprendizaje (load_learner(path_model, nombre_learner),
                                            nombre_learner, descripcion_learner))
        except RuntimeError as e:
            if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
                print(e)
                message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
                raise RuntimeError(message)
            else:
                raise
    return


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
lista_learn = []
loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img_pil = PIL.Image.open(BytesIO(img_bytes))
    img_w, img_h = img_pil.size
    if min(img_w, img_h) > 322:
        ratio = 322 / min(img_w, img_h)
        img_pil = img_pil.resize((int(img_w * ratio), int(img_h * ratio)),
                                 resample=PIL.Image.BILINEAR).convert('RGB')
    img = Image(pil2tensor(img_pil.convert("RGB"), np.float32).div_(255))
    prediccion = ''
    for aprender in lista_learn:
        pred_clase, pred_idx, salida = aprender.learner.predict(img)
        prediccion += " %s %s: %.2f %% \n" % (aprender.descripcion, pred_clase, salida[pred_idx] * 100)
    return JSONResponse({'result': prediccion})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        puerto = int(sys.argv[2])
        uvicorn.run(app=app, host='0.0.0.0', port=puerto, log_level="info")
