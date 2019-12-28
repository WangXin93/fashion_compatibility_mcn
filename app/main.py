import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import time
import os
import json
import sys
import torch
import re

sys.path.insert(0, "../mcn")
import torchvision.transforms as transforms
from model import CompatModel
from utils import prepare_dataloaders
from PIL import Image

train_dataset, _, _, _, test_dataset, _ = prepare_dataloaders(num_workers=1)
# Load pretrained weights
device = torch.device('cpu')
# print(len(.vocabulary)) # 2757
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=2757).to(device)
model.load_state_dict(torch.load("../mcn/model_train_relation_vse_type_cond_scales.pth", map_location="cpu"))
model.eval()
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

def defect_detect(img, model, normalize=True):
    # Register hook for comparison matrix
    relation = None

    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        if name == 'fc1':
            module.register_backward_hook(func_r)
    # Forward
    out  = model._compute_score(img)
    out = out[0]

    # Backward
    one_hot = torch.FloatTensor([[-1]]).to(device)
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)

    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

def item_diagnosis(relation, select):
    """ Output the most incompatible item in the outfit
    
    Return:
        result (list): Diagnosis value of each item 
        order (list): The indices of items ordered by its importance
    """
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape).byte()
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

def vec2mat(relation, select):
    """ Convert relation vector to 4 matrix, which is corresponding to 4 layers
    in the backend CNN.
    
    Args:
        relation: (np.array | torch.tensor) of shpae (60,)
        select: List of select item indices, e.g. (0, 2, 3) means select 3 items
            in total 5 items in the outfit.
        
    Return:
        mats: List of matrix
    """
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

root = "/home/wangx/fashion_compatibility_mcn/data"
img_root = os.path.join(root, "images")
json_file = os.path.join(root, "test_no_dup_with_category_3more_name.json")

json_data = json.load(open(json_file))

top_options, bottom_options, shoe_options, bag_options, accessory_options = [], [], [], [], []
print("Load options...")
for cnt, (iid, outfit) in enumerate(json_data.items()):
    if cnt > 10:
        break
    if "upper" in outfit:
        label = os.path.join(iid, str(outfit['upper']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        top_options.append({'label': label, 'value': value})
    if "bottom" in outfit:
        label = os.path.join(iid, str(outfit['bottom']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        bottom_options.append({'label': label, 'value': value})
    if "shoe" in outfit:
        label = os.path.join(iid, str(outfit['shoe']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        shoe_options.append({'label': label, 'value': value})
    if "bag" in outfit:
        label = os.path.join(iid, str(outfit['bag']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        bag_options.append({'label': label, 'value': value})
    if "accessory" in outfit:
        label = os.path.join(iid, str(outfit['accessory']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        accessory_options.append({'label': label, 'value': value})


app.layout = html.Div([
    html.H1("Fashion Outfit Diagnosis", style={
        "margin": "0.5em 1em 0.5em 1em"
    }),
    html.Div([
        html.Div([
            html.H4(children="Top"),
            dcc.Dropdown(
                id='top',
                options=top_options,
                value=random.choice(top_options)['value'],
                style={"float": "left", "width": "300px"}
            ),
            dcc.Upload(id="upload-top", children=['Drop here or ', html.A('Upload')], style={
                "margin-left": "300px", "textAlign": "center", "border": "1px dashed black", "line-height": "34px", "height": "34px", "border-radius": "5px"
            }),
        ], style={"margin": "1em 1em 1em 1em"}),
        html.Div([
            html.H4(children="bottom"),
            dcc.Dropdown(
                id='bottom',
                options=bottom_options,
                value=random.choice(bottom_options)['value'],
                style={"float": "left", "width": "300px"}
            ),
            dcc.Upload(id="upload-bottom", children=['Drop here or ', html.A('Upload')], style={
                "margin-left": "300px", "textAlign": "center", "border": "1px dashed black", "line-height": "34px", "height": "34px", "border-radius": "5px"
            }),
        ], style={"margin": "1em 1em 1em 1em"}),
        html.Div([
            html.H4(children="shoe"),
            dcc.Dropdown(
                id='shoe',
                options=shoe_options,
                value=random.choice(shoe_options)['value'],
                style={"float": "left", "width": "300px"}
            ),
            dcc.Upload(id="upload-shoe", children=['Drop here or ', html.A('Upload')], style={
                "margin-left": "300px", "textAlign": "center", "border": "1px dashed black", "line-height": "34px", "height": "34px", "border-radius": "5px"
            }),
        ], style={"margin": "1em 1em 1em 1em"}),
        html.Div([
            html.H4(children="bag"),
            dcc.Dropdown(
                id='bag',
                options=bag_options,
                value=random.choice(bag_options)['value'],
                style={"float": "left", "width": "300px"}
            ),
            dcc.Upload(id="upload-bag", children=['Drop here or ', html.A('Upload')], style={
                "margin-left": "300px", "textAlign": "center", "border": "1px dashed black", "line-height": "34px", "height": "34px", "border-radius": "5px"
            }),
        ], style={"margin": "1em 1em 1em 1em"}),
        html.Div([
            html.H4(children="accessory"),
            dcc.Dropdown(
                id='accessory',
                options=accessory_options,
                value=random.choice(accessory_options)['value'],
                style={"float": "left", "width": "300px"}
            ),
            dcc.Upload(id="upload-accessory", children=['Drop here or ', html.A('Upload')], style={
                "margin-left": "300px", "textAlign": "center", "border": "1px dashed black", "line-height": "34px", "height": "34px", "border-radius": "5px"
            }),
        ], style={"margin": "1em 1em 1em 1em"}),
        html.Button(id='submit-button', n_clicks=0, children='Submit', style={
            "margin": "1.5em"
        }),
    ], style={
        "display": "inline-block",
        "vertical-align": "top",
        "width": "35%",
        "border": "1px solid black",
        "border-radius": "5px",
    }),
    html.Div([
        html.Div(id="input-state", children=[
            html.H4(children="Current outfit"),
            html.Img(id='top-img', style={"max-height":"150px", "margin":"5px"}),
            html.Img(id='bottom-img', style={"max-height":"150px", "margin": "5px"}),
            html.Img(id='shoe-img', style={"max-height": "150px", "margin": "5px"}),
            html.Img(id='bag-img', style={"max-height": "150px", "margin": "5px"}),
            html.Img(id='accessory-img', style={"max-height": "150px", "margin": "5px"}),
        ]),
        html.Div(id="output-state")
    ], style={
        "display": "inline-block",
        "vertical-align": "top",
        "width": "60%",
        "margin": "1em 1em 1em 1em",
    })
])

@app.callback(
    Output('top-img', 'src'),
    [Input('top', 'value'), Input('upload-top', 'contents')],
    [State('upload-top', 'filename'),
     State('upload-top', 'last_modified')])
def update_top(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('bottom-img', 'src'),
    [Input('bottom', 'value'), Input('upload-bottom', 'contents')],
    [State('upload-bottom', 'filename'),
     State('upload-bottom', 'last_modified')])
def update_bottom(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('shoe-img', 'src'),
    [Input('shoe', 'value'), Input('upload-shoe', 'contents')],
    [State('upload-shoe', 'filename'),
     State('upload-shoe', 'last_modified')])
def update_shoe(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('bag-img', 'src'),
    [Input('bag', 'value'), Input('upload-bag', 'contents')],
    [State('upload-bag', 'filename'),
     State('upload-bag', 'last_modified')])
def update_bag(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('accessory-img', 'src'),
    [Input('accessory', 'value'), Input('upload-accessory', 'contents')],
    [State('upload-accessory', 'filename'),
     State('upload-accessory', 'last_modified')])
def update_accessory(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(Output('output-state', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('top-img', 'src'),
               State('bottom-img', 'src'),
               State('shoe-img', 'src'),
               State('bag-img', 'src'),
               State('accessory-img', 'src')])
def update_output(n_clicks, top, bottom, shoe, bag, accessory):
    if n_clicks > 0:
        img_dict = {
            "top": top.split(',')[1],
            "bottom": bottom.split(',')[1],
            "shoe": shoe.split(',')[1],
            "bag": bag.split(',')[1],
            "accessory": accessory.split(',')[1]
        }
        img_tensor = base64_to_tensor(img_dict)
        img_tensor.unsqueeze_(0)
        relation, score = defect_detect(img_tensor, model)
        relation = relation.squeeze()
        result, order = item_diagnosis(relation, select=[0, 1, 2, 3, 4])
        best_score, best_img_path = retrieve_sub(img_tensor, [0, 1, 2, 3, 4], order)

        out = [html.H4(children="Result"), html.H4(children="Score: {:.4f}".format(score)), html.H4(children="Revised Score: {:.4f}".format(best_score))]

        for part in ["top", "bottom", "shoe", "bag", "accessory"]:
            if part in best_img_path.keys():
                fname = best_img_path[part]
                encoded_img = base64.b64encode(open(fname, "rb").read())
                src= 'data:image/png;base64,{}'.format(encoded_img.decode())
            else:
                src = locals()[part]
            out.append(html.Img(id='{}-img-new'.format(part), style={"max-height":"150px", "margin":"5px"}, src=src))

        return out

def item_diagnosis(relation, select):
    """ Output the most incompatible item in the outfit
    
    Return:
        result (list): Diagnosis value of each item 
        order (list): The indices of items ordered by its importance
    """
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape).byte()
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

def retrieve_sub(x, select, order):
    """ Retrieve the datset to substitute the worst item for the best choice.
    """
    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}
    try_most = 20
   
    best_score = -1
    best_img_path = dict()

    for o in order:
        if best_score > 0.9:
            break
        problem_part_idx = select[o]
        problem_part = all_names[problem_part_idx]
        for outfit in random.sample(test_dataset.data, try_most):
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img).to(device)
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    out = model._compute_score(x)
                    score = out[0]
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
    
        print('problem_part: {}'.format(problem_part))
        print('best substitution: {} {}'.format(problem_part, best_img_path[problem_part]))
        print('After substitution the score is {:.4f}'.format(best_score))
    return best_score, best_img_path

def base64_to_tensor(image_bytes_dict):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    outfit_tensor = []
    for k, v in image_bytes_dict.items():
        img = base64_to_image(v)
        tensor = my_transforms(img)
        outfit_tensor.append(tensor.squeeze())
    outfit_tensor = torch.stack(outfit_tensor)
    outfit_tensor = outfit_tensor.to(device)
    return outfit_tensor

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    return img

if __name__ == "__main__":
    app.run_server(debug=True)

