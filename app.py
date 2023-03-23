from os import path
import os
from flask import send_from_directory, Flask, render_template, request, app, url_for, redirect,flash, session,send_from_directory, current_app, jsonify
import numpy
import lightning.pytorch as pl
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import torch.nn as nn
import sqlite3
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

app = Flask(__name__)
app.secret_key = 'f3167129525f2a20696b7de80ff37401c963b55871119ed7ddec510809d5fa5530fa40bdf5041484a52a3932a4cad6e542e3c5199ef4cca9aa7c7e52f69c3e76'

with sqlite3.connect("pqrs.db") as con:
  con.row_factory = sqlite3.Row
  cur = con.cursor()
  cur.execute("CREATE TABLE if not exists pqrs(id INTEGER PRIMARY KEY AUTOINCREMENT, nombres text NOT NULL, apellidos text NOT NULL, correo text NOT NULL, numerotlf text NOT NULL, mensaje text NOT NULL, clasificacion_deep_learning text, clasificacion_svm text)")
  con.commit()

@app.route('/', methods=['GET'])
def Inicio():
  return render_template('pqrMain.html')

@app.route('/consult-all', methods=['GET'])
def GetAll():
  with sqlite3.connect("pqrs.db") as con:
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    res = cur.execute("SELECT * FROM pqrs")
    res = res.fetchall()
    res=[dict(o) for o in res]
  return jsonify(res)

@app.route('/pqrs-process', methods=['GET','POST'])
def PqrsProcess():
  if request.method == "GET":
    return "method get unavailable"
  
  if request.method == 'POST':
    dataUser = request.json
    nombres = dataUser['nombres']
    apellidos = dataUser['apellidos']
    email = dataUser['email']
    telfContacto = dataUser['telefono']
    comment = dataUser['comentario']
    chk_path = os.path.join(os.getcwd(),"best-checkpoint.ckpt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
    tokenizerBart = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt") #por si acaso se puede pasar este parametro tambien use_fast = False
    tokenizerBart.src_lang = "es_XX"
    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    LABEL_COLUMNS = ['con_int_discipli',
    'des_alcalde',
    'direccion',
    'espacio_publico',
    'extra',
    'ofc_tic',
    'pla_sisben',
    'prensa_y_comuni',
    'sec_admin',
    'sec_des_social',
    'sec_educacion',
    'sec_hacienda',
    'sec_infraest',
    'sec_interior',
    'sec_juridica',
    'sec_planeacion',
    'sec_salud_y_ambi',
    'ser_publicos',
    'st',
    'valorizacion']
    class SecretariasClassificationModel(pl.LightningModule):
      def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.n_classes = n_classes
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.n_classes) #this are our linear layer to classify
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.training_step_outputs = []
        
      def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
          loss = self.criterion(output, labels)
        return loss, output
      def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.training_step_outputs.append(outputs)
        return {"loss": loss, "predictions": outputs, "labels": labels}
      def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
      def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
      def on_train_epoch_end(self):
        labels = []
        predictions = []
        outputs = self.training_step_outputs
        for output in outputs:
          for out_labels in output[0].cpu():
            labels.append(out_labels)
          for out_predictions in output[1].cpu():
            predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        num_classes=self.n_classes
        auroc = AUROC(task="binary")
        class_roc_auc = auroc(predictions, labels)
        self.logger.experiment.add_scalar(f"auroc/Train", class_roc_auc, self.current_epoch)
      def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
        )
        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )

    trained_model = SecretariasClassificationModel.load_from_checkpoint(
        chk_path,
        n_classes=len(LABEL_COLUMNS)
        )
    trained_model.eval()
    trained_model.freeze()

    encoded_text = tokenizerBart(comment, return_tensors="pt", max_length=512, truncation=True)
    generated_tokens = model.generate(**encoded_text)
    comment_english = tokenizerBart.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    encoding = tokenizer.encode_plus(
    comment_english,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding="max_length",
    return_attention_mask=True,
    return_tensors='pt',
    )
    _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()
    secretariasDict = {}
    for label, prediction in zip(LABEL_COLUMNS, test_prediction):
        secretariasDict[label] = prediction
    max_val = max(secretariasDict.values())
    max_val_secretaria = max(secretariasDict, key=secretariasDict.get)
    val_rfc = None
    try:
      tfidfVectorizer = load('tfidf.joblib')
      #encoded_comment=tfidfVectorizer.transform([comment])
      rfc = load('random_forest_classifier.joblib')
      val_rfc = rfc.predict(tfidfVectorizer.transform([comment]))
    except:
      val_rfc = max_val_secretaria
    with sqlite3.connect("pqrs.db") as con:
      cur = con.cursor()
      cur.execute("""
          INSERT INTO pqrs (nombres,apellidos,correo,numerotlf,mensaje,clasificacion_deep_learning, clasificacion_svm) VALUES
              (?, ?, ?, ?, ?, ?, ?)
      """, (nombres,apellidos,email,telfContacto,comment,max_val_secretaria,val_rfc))
      con.commit()
    resp = jsonify(success=True)
    return resp

if __name__ == '__main__':
    app.run(debug=True, port=5001)