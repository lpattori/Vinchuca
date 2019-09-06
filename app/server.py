import aiohttp
import asyncio

import uvicorn
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from app.heatmaps.heatmap import heatmap



class Aprendizaje():
    "Structure  to  save learner name and descriptión"
    def __init__(self, learner: Learner, name: str, description: str):
        self.learner = learner
        self.nombre = name
        self.description = description


csv.register_dialect('no_quotes', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)
csv_file_url = ('https://onedrive.live.com/download?'
                'cid=27B3CFFF6EE897C2&resid=27B3CFFF6EE897C2%2122311&authkey=AFQ1XNRERUjj8eI')
csv_file_name = 'parametros.csv'
path = Path(__file__).parent
path_model = path / 'models'
path_img = path / 'static'
global_img : Image

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#app.mount('/static', StaticFiles(directory='app/static'))
app.mount('/static', StaticFiles(directory='static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(csv_file_url, path_model / csv_file_name)
    with open(path_model / csv_file_name, 'r') as description:
        reader = csv.reader(description, dialect='no_quotes')
        lista_redes = list(reader)
    for learner_name, learner_desription, onedrive_url in lista_redes:
        await download_file(onedrive_url, path_model / learner_name)
        try:
            lista_learn.append(Aprendizaje (load_learner(path_model, learner_name),
                                            learner_name, learner_desription))
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
    global global_img
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img_pil = PIL.Image.open(BytesIO(img_bytes))
    img_w, img_h = img_pil.size
    if min(img_w, img_h) > 322:
        ratio = 322 / min(img_w, img_h)
        img_pil = img_pil.resize((int(img_w * ratio), int(img_h * ratio)),
                                 resample=PIL.Image.BILINEAR).convert('RGB')
    global_img = Image(pil2tensor(img_pil.convert("RGB"), np.float32).div_(255))
    prediction = []
    for aprender in lista_learn:
        pred_clase, pred_idx, certezas = aprender.learner.predict(global_img)
        n = 3
        mejores = np.argsort(certezas.numpy())[::-1][:n] # take n better results
        clases = []
        for i in mejores:
            clases.append((aprender.learner.data.classes[i], "%.2f%%" % (certezas[i]*100), " %d" % i))
        prediction.append((aprender.description, clases))
    return JSONResponse(prediction)

@app.route('/heat/', methods=['POST'])
async def heat(request):
    calor = await request.json()
    learner = int(calor['learner'])
    clase = int(calor['clase'])
    first = calor['first']
    img_heat = heatmap(lista_learn[learner].learner, global_img, clase, first)
    with io.BytesIO() as contenido:
        img_heat.save(contenido, format="JPEG")
        crudo = contenido.getvalue()
    return StreamingResponse(BytesIO(crudo), media_type='image/jpeg')

if __name__ == '__main__':
    if 'serve' in sys.argv:
        puerto = int(sys.argv[2])
        uvicorn.run(app=app, host='0.0.0.0', port=puerto, log_level="info")
    else:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

