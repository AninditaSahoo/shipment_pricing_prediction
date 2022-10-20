from flask import Flask
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import pcolor
import pandas as pd

app=Flask(__name__,template_folder="templates")
cluster0=pickle.load(open("model_cluster0.pkl","rb"))
cluster1=pickle.load(open("model_cluster1.pkl","rb"))
scaler=pickle.load(open("preprocessed.pkl","rb"))

@app.route('/',methods=['POST', 'GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    brand=request.form.get("brand")
    country=request.form.get("country")
    dosage_form=request.form.get("dosage_form")
    first_line_designation=request.form.get("first_line_designation")
    fulfill_via=request.form.get("fulfill_via")
    managed_by=request.form.get("managed_by")
    product_group=request.form.get("product_group")
    shipment_mode=request.form.get("shipment_mode")
    sub_classification=request.form.get("sub_classification")
    dosage=request.form.get("dosage")
    freight_cost=request.form.get("freight_cost")
    line_item_insurance=request.form.get("line_item_insurance")
    line_item_quantity=request.form.get("line_item_quantity")
    line_item_value=request.form.get("line_item_value")
    pack_price=request.form.get("pack_price")
    unit_of_measure=request.form.get("unit_of_measure")
    unit_price=request.form.get("unit_price")
    weight=request.form.get("weight")


    df=pd.DataFrame({"brand":[brand],"country":[country],"dosage_form":[dosage_form],"first_line_designation":[first_line_designation],"fulfill_via":[fulfill_via],"managed_by":[managed_by],"product_group":[product_group],"shipment_mode":[shipment_mode],
    "sub_classification":[sub_classification],"dosage":[dosage],"freight_cost":[freight_cost],"line_item_insurance":[line_item_insurance],"line_item_quantity":[line_item_quantity],"line_item_value":[line_item_value],"pack_price":[pack_price],"unit_of_measure":[unit_of_measure],"unit_price":[unit_price],"weight":[weight]})
    df1=pd.DataFrame(scaler.transform(df),columns=['brand_Uni-Gold','brand_Others',"brand_Kaletra","brand_Determine","brand_Aluvia",'country_Others', 'country_Mozambique', 'country_South Africa', 'country_CÃte dIvoire', 'country_Nigeria', 'country_Zimbabwe', 'country_Uganda', 'country_Vietnam', 'country_Rwanda', 'country_Haiti', 'country_Tanzania','dosage_form_Tablet - FDC', 'dosage_form_Test kit', 'dosage_form_Oral solution', 'dosage_form_Capsule', 'dosage_form_Others','first_line_designation_Yes','fulfill_via_Direct Drop','managed_by_Haiti Field Office', 'managed_by_Ethiopia Field Office','product_group_HRDT', 'product_group_ANTM', 'product_group_MRDT', 'product_group_ACT','shipment_mode_Air', 'shipment_mode_Ocean', 'shipment_mode_Air Charter','sub_classification_HIV test', 'sub_classification_Pediatric', 'sub_classification_Malaria', 'sub_classification_HIV test - Ancillary', 'sub_classification_ACT',"dosage","freight_cost","line_item_insurance","line_item_quantity","line_item_value","pack_price","unit_of_measure","unit_price","weight"])
    df1.drop(["line_item_value"],axis=1,inplace=True)
    ans=cluster0.predict(df1)
    

    return render_template('index.html', prediction_text='Shipping price will be {}'.format(ans))



if __name__=="__main__":
    app.run(debug=True)