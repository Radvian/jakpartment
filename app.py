import streamlit as st
# Essentials
import numpy as np
import pandas as pd
import datetime
import random

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
import xgboost
from xgboost.sklearn import XGBRegressor
import lightgbm
from lightgbm import LGBMRegressor

# Misc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import pickle
import joblib
import folium
import branca.colormap as cm
from streamlit_folium import folium_static
import bs4
from bs4 import BeautifulSoup as bs
import requests
import json
import re
import base64

def main():
    pd.set_option('display.max_colwidth', None)
    ### Dataset Import and Dashboard DataFrame Preparation (renaming columns for aesthetic purposes)
    cleaned = pd.read_csv(r'Web Scraping and Dataset\\Cleaned Apartment Data.csv')

    UnitType = list()
    for _ in cleaned.No_Rooms:
        if _ == 0:
            UnitType.append('Studio')
        elif _ == 1:
            UnitType.append('1BR')
        elif _ == 2:
            UnitType.append('2BR')
        elif _ == 3:
            UnitType.append('3BR')
        elif _ == 4:
            UnitType.append('4+BR')

    FStatus = list()
    for f in cleaned.Furnished:
        if f == 0:
            FStatus.append('Non Furnished')
        else:
            FStatus.append('Fully Furnished')
    
    cleaned = cleaned.rename({'Locality':'District', 'Water_Heater':'Water Heater', 'Dining_Set':'Dining Set',
               'Access_Card':'Access Card', 'Kitchen':'Kitchen Set', 'Fridge':'Refrigerator', 
                'Washing_Machine':'Washing Machine', 'TV_Cable':'TV Cable', 'Grocery':'Grocery Shop', 
                'Internet':'Internet Services', 'Swim_Pool':'Swimming Pool', 'Basketball':'Basketball Field',
               'Multipurpose_Room':'Multipurpose Room', 'Jogging':'Jogging Track', 'Tennis':'Tennis Field', 
                'Playground':'Kids Playground', 'Total_Facilities':'Total Number of Facilities', 
                'AnnualPrice':'Annual Price'}, axis='columns')

    ML_Ready = cleaned #dataset to be used on the ML page

    dash = cleaned #dataset to be used on the visualization page
    dash['Unit Type'] = UnitType
    dash['Furnished Status'] = FStatus
    dash = dash.drop(['Unnamed: 0', 'No_Rooms', 'Furnished', 'FurnishedNew'], axis = 'columns')

    dash = dash[['Unit Type', 'Bathroom', 'Region', 'District', 'Longitude', 'Latitude', 'Furnished Status',
        'Area', 'AC', 'Water Heater', 'Dining Set', 'Bed', 'Kitchen Set', 'Refrigerator', 'Washing Machine', 
        'TV', 'ATM', 'TV Cable', 'Grocery Shop', 'Internet Services', 'Swimming Pool', 'Basketball Field',
        'Multipurpose Room', 'Jogging Track', 'Tennis Field', 'Kids Playground', 
        'Total Number of Facilities', 'Annual Price']]


    ML_Ready = ML_Ready.drop(['FurnishedNew'], axis='columns')
    ML_Ready = ML_Ready.rename({'No_Rooms':'Number of Bedrooms'}, axis = 'columns')
    InJakCheck = list()
    for reg in ML_Ready.Region:
        if 'Jakarta' in reg:
            InJakCheck.append(1)
        else:
            InJakCheck.append(0)
    
    ML_Ready['In-Jakarta Check'] = InJakCheck
    ###    
    st.sidebar.title('Navigation')
    pages = st.sidebar.radio("Pages", ("Home Page", "Apartment Rent Price Calculator", "Web Scraping Demo", 
    "Data Visualization", "Play with Machine Learning Models", "Summary", "FAQs", "About the Author"), index = 0)
    if pages == "Home Page":

        st.title('Welcome to the Jakpartment Project!')
        st.image('images\\apartment.jpg', width = 650)
        st.write("Open the navigation sidebar and select any pages to proceed. Happy exploring!")
    
    elif pages == "Apartment Rent Price Calculator":

        st.title("Jabodetabek Apartment Annual Rent Price Predictor")
        st.markdown("Enter your desired apartment unit and we'll estimate the annual rent price.")

        #Load Location Dictionary
        load_dict = open('Temp Files\location_dict.pkl', 'rb')
        location = pickle.load(load_dict)

        #Define functions to find longitude/latitude
        def lon_finder(dict, region, locality):
            return float(dict[region][locality].split(',')[0])

        def lat_finder(dict, region, locality):
            return float(dict[region][locality].split(',')[1])

        #Make empty lists of features
        No_Rooms = list()
        Bathroom = list() 
        Longitude = list() 
        Latitude= list() 
        Furnished = list() 
        Area= list() 
        AC= list()
        Water_Heater= list() 
        Dining_Set= list() 
        Bed= list() 
        Kitchen= list() 
        Fridge= list()
        Washing_Machine= list() 
        TV= list()
        ATM= list() 
        TV_Cable= list() 
        Grocery= list() 
        Internet= list()
        Swim_Pool= list() 
        Laundry= list()
        Security= list() 
        Basketball= list() 
        Multipurpose_Room= list()
        Gym= list()
        Jogging= list() 
        Tennis= list() 
        Restaurant= list() 
        Playground= list() 
        Jakcheck= list()
        Total_Facilities=list()

        st.subheader("What's your apartment unit type?")
        unit_type = st.selectbox("Unit Type",("Studio", "1 Bedroom(s)", "2 Bedroom(s)", "3 Bedroom(s)", 
                                "4 (or more) Bedroom(s)"))
        st.subheader('Is your apartment unit fully furnished?')
        furnished = st.radio("",("Yes", "No"))

        if furnished == "Yes":
            Furnished.append(1)  #Fully furnished
        else:
            Furnished.append(0)  #Non furnished
    
        st.subheader("How much is your apartment unit's area?")
        if unit_type == "Studio":
            No_Rooms.append(0)
            st.write('Studio apartment units have area ranging from 20 - 100 m\u00b2 with an average of 27 m\u00b2.')
            area = st.slider('Area', 20, 100)
            Area.append(area)
        elif unit_type == '1 Bedroom(s)':
            No_Rooms.append(1)
            st.write('1 Bedroom(s) apartment units have area ranging from 21 - 129 m\u00b2 with an average of 45 m\u00b2.')
            area = st.slider('Area', 21, 129)
            Area.append(area)
        elif unit_type == '2 Bedroom(s)':
            No_Rooms.append(2)
            st.write('2 Bedroom(s) apartment units have area ranging from 28 - 232 m\u00b2 with an average of 59 m\u00b2.')
            area = st.slider('Area', 28, 232)
            Area.append(area)
        elif unit_type == '3 Bedroom(s)':
            No_Rooms.append(3)
            st.write('3 Bedroom(s) apartment units have area ranging from 38 - 250 m\u00b2 with an average of 121 m\u00b2.')
            area = st.slider('Area', 38, 250)
            Area.append(area)
        elif unit_type == '4 (or more) Bedroom(s)':
            No_Rooms.append(4)
            st.write('4 Bedroom(s) apartment units have area ranging from 92 - 250 m\u00b2 with an average of 211 m\u00b2.')
            area = st.slider('Area', 92, 250)
            Area.append(area)
        
        st.subheader('How many bathrooms are there?')
        bathroom = st.selectbox("Number of bathroom(s)", (1,2,3,4,5))

        st.subheader("Where is your apartment unit? (Region)")
        region = st.selectbox("Region", ("Jakarta Utara", 'Jakarta Barat', "Jakarta Pusat", "Jakarta Selatan", "Jakarta Timur", 
                "Bogor", "Depok", "Tangerang", "Bekasi"))
        if region == "Jakarta Utara":
            Jakcheck.append(1)
            locality = st.selectbox("District", ("Ancol", "Kelapa Gading", "Pantai Indah Kapuk", "Pluit", "Sunter"))
        elif region == "Jakarta Barat":
            Jakcheck.append(1)
            locality = st.selectbox("District", ("Cengkareng", "Daan Mogot", "Duri Kosambi", "Gajah Mada", "Grogol",
                    "Kalideres", "Kebon Jeruk", "Kedoya", "Kembangan", "Palmerah", "Pos Pengumben", "Puri Indah",
                    "Slipi", "Taman Sari", "Tanjung Duren"))
        elif region == "Jakarta Selatan":
            Jakcheck.append(1)
            locality = st.selectbox("District", ("Bintaro", "Casablanca", "Cilandak", "Dharmawangsa", "Epicentrum",
                    "Fatmawati", "Gandaria", "Gatot Subroto", "Kalibata", "Kebagusan", "Kebayoran Baru", 
                    "Kebayoran Lama", "Kemang", "Kuningan", "Lebak Bulus", "Mega Kuningan", "Pakubuwono",
                    "Pancoran", "Pasar Minggu", "Pejaten", "Permata Hijau", "Pesanggrahan", "Pondok Indah",
                    "Radio Dalam", "Rasuna Said", "SCBD", "Semanggi", "Senayan", "Senopati", "Setiabudi", "Simprug",
                    "Sudirman", "TB Simatupang", "Tebet"))
        elif region == "Jakarta Pusat":
            Jakcheck.append(1)
            locality = st.selectbox("District", ("Bendungan Hilir", "Cempaka Putih", "Gatot Subroto", "Gunung Sahari", 
                    "Kemayoran", "Mangga Besar", "Mangga Dua", "Menteng", "Pasar Baru", "Pecenongan", "Salemba", 
                    "Senayan", "Senen", "Sudirman", "Tanah Abang", "Thamrin"))
        elif region == "Jakarta Timur":
            Jakcheck.append(1)
            locality = st.selectbox("District", ("Cakung", "Cawang", "Cibubur", "Cipinang", "Jatinegara", "Kampung Melayu",
                    "Kelapa Gading", "MT Haryono", "Pasar Rebo", "Pondok Bambu", "Pulomas"))
        elif region == "Bogor":
            Jakcheck.append(0)
            locality = st.selectbox("District", ("Sentul", "Tanah Sareal"))
        elif region == "Depok":
            Jakcheck.append(0)
            locality = st.selectbox("District", ("Cimanggis", "Cinere", "Margonda"))
        elif region == "Tangerang":
            Jakcheck.append(0)
            locality = st.selectbox("District", ("Alam Sutera", "BSD City", "Bintaro", "Cengkareng", "Cikokol", "Cipondoh",
                    "Ciputat", "Daan Mogot", "Gading Serpong", "Karang Tengah", "Karawaci", "Kelapa Dua - Tanggerang", 
                    "Serpong"))
        elif region == "Bekasi":
            Jakcheck.append(0)
            locality = st.selectbox("District", ("Bekasi", "Bekasi Timur", "Cikarang", "Kalimalang", "Lippo Cikarang", 
                    "Pekayon", "Summarecon Bekasi"))

        #Map region and district (locality) input to numerical coordinate
        Longitude.append(lon_finder(location, region, locality)) 
        Latitude.append(lat_finder(location, region, locality))

        st.subheader("Which in-room facilities that your unit have?")

        def facil(displayname, featurename):
            if st.checkbox(displayname):
                featurename.append(1)
            else:
                featurename.append(0)

        facil('Air Conditioner', AC)
        facil('Water Heater', Water_Heater)
        facil('Dining Set', Dining_Set)
        facil('Bed', Bed)
        facil('Kitchen Set', Kitchen)
        facil('Refrigerator', Fridge)
        facil('Washing Machine', Washing_Machine)
        facil('TV', TV)
        
        st.subheader('Which apartment facilities are present?')

        facil('ATM', ATM)
        facil('TV Cable Services', TV_Cable)
        facil('Grocery Shops', Grocery)
        facil('Internet Services', Internet)
        facil('Clothing Laundry', Laundry)
        facil('Security (CCTV, etc)', Security)
        facil('Restaurant', Restaurant)
        facil('Multipurpose Room', Multipurpose_Room)
        facil('Kids Playground', Playground)

        st.subheader('Which sports facilities are present?')
        facil('Swimming Pool', Swim_Pool)
        facil('Basketball Field', Basketball)
        facil('Gym', Gym)
        facil('Jogging Track', Jogging)
        facil('Tennis Field', Tennis)
        
        #total facilities
        total = AC[0] + Water_Heater[0] + Dining_Set[0] + Bed[0] + Kitchen[0] + Fridge[0] + Washing_Machine[0] + \
            TV[0] + ATM[0] + TV_Cable[0] + Grocery[0] + Internet[0] + Laundry[0] + Security[0] + Multipurpose_Room[0] + \
            Restaurant[0] + Playground[0] + Swim_Pool[0] + Basketball[0] + Gym[0] + Jogging[0] + Tennis[0]
        Total_Facilities.append(total)


        df = pd.DataFrame({'No_Rooms':No_Rooms, 'Bathroom': bathroom, 'Longitude':Longitude, 'Latitude':Latitude, 'Furnished':Furnished, 'Area':Area, 'AC':AC,
       'Water_Heater':Water_Heater, 'Dining_Set':Dining_Set, 'Bed':Bed, 'Kitchen':Kitchen, 'Fridge':Fridge,
       'Washing_Machine':Washing_Machine, 'TV':TV, 'ATM':ATM, 'TV_Cable':TV_Cable, 'Grocery':Grocery, 'Internet':Internet,
       'Swim_Pool':Swim_Pool, 'Laundry':Laundry, 'Security':Security, 'Basketball':Basketball, 'Multipurpose_Room':Multipurpose_Room,
       'Gym':Gym, 'Jogging':Jogging, 'Tennis':Tennis, 'Restaurant':Restaurant, 'Playground':Playground, 'Jakcheck':Jakcheck, 'Total_Facilities':Total_Facilities
            })

        if st.button("Calculate Price", key='calculate'):
            lgb = joblib.load('Models\lgb_final.joblib.dat')
            price = int(lgb.predict(df)[0])
            
            lgb_001 = joblib.load('Models\lgb_001.joblib.dat')
            lower_price1 = int(lgb_001.predict(df)[0])

            lgb_099 = joblib.load('Models\lgb_099.joblib.dat')
            upper_price99 = int(lgb_099.predict(df)[0])

            lgb_005 = joblib.load('Models\lgb_005.joblib.dat')
            lower_price5 = int(lgb_005.predict(df)[0])

            lgb_095 = joblib.load('Models\lgb_095.joblib.dat')
            upper_price95 = int(lgb_095.predict(df)[0])

            str_price = format(price, ',')
            str_lowprice1 = format(lower_price1, ',')
            str_upprice99 = format(upper_price99, ',')

            str_lowprice5 = format(lower_price5, ',')
            str_upprice95 = format(upper_price95, ',')

            st.subheader("Your apartment unit's annual rent price is predicted at IDR {}".format(str_price))
            st.subheader("A 99% prediction interval of your unit's price is IDR {} until IDR {}".format(str_lowprice1, str_upprice99))
            st.subheader("A 95% prediction interval of your unit's price is IDR {} until IDR {}".format(str_lowprice5, str_upprice95))

        st.subheader("Author's note")
        st.write("If your apartment unit is located on top of a mall (or within walking distance to a mall), it is reasonable that the price might be \
                higher than what is predicted here. If your apartment unit is marketed in a special promotion event, the price might be lower than what is predicted here")
        st.subheader("Learn more about how the price is calculated")
        st.write("The predicted price as well as the prediction interval is calculated by inputing the apartment unit details above into a saved Light GBM Regressor model. \
            The model, when fitted to the scraped dataset, achieved around 0.9 R-squared score.")
        
    elif pages == "Web Scraping Demo":
        st.title('Simple Web Scraping Demo')
        st.subheader("Welcome to the web scraping demo page!")
        st.write("In this page, we'll simulate how an apartment unit's page is scraped during my data acquisition phase of this project.")
        st.markdown("To use this scraper, insert a link of an apartment unit from [Jendela 360](https://www.jendela360.com) website.")
        st.markdown("Here's a short GIF on how you can find an apartment unit page in Jendela 360.")
        gif_view = st.radio("Toggle GIF viewing option:", ("View GIF", "Hide GIF"), index = 0)
        if gif_view == "View GIF":
            file_ = open("images\scrape_example.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="scrape_example_gif">',
                unsafe_allow_html=True,
            )

        url = st.text_input('Enter the link here:')

        def get_ld_json(url: str) -> dict:
            parser = "html.parser"
            req = requests.get(url)
            soup = bs(req.text, parser)
            return json.loads("".join(soup.find("script", {"type":"application/ld+json"}).contents))

        def main_feature_extractor(feature):
            feature_soup = soup.find("ul",{"class":'gridded--list bordered--grid'})
            if feature_soup is None:
                feature_soup = feature_soup = soup.find("ul",{"class":'gridded--list'})
            feature = feature_soup.findChild("img",{"alt":feature}).find_parent("li").get_text().strip()
            return(feature)
    
        def facility_checker(facility):
            if facility in facilities:
                return(1)
            else:
                return(0)

        if st.button('Scrape it!'):
            try:
                r = requests.get(url)
                soup = bs(r.content)
    
                ##Apartment Metadata
                apartment_metadata = get_ld_json(url)
                Apt_Name = apartment_metadata['name']
    
                No_Rooms = apartment_metadata['numberOfRooms']
    
                Street = apartment_metadata['address']['streetAddress']
    
                Locality = apartment_metadata['address']['addressLocality']
    
                Region = apartment_metadata['address']['addressRegion']
    
                Longitude = apartment_metadata['geo']['longitude']
    
                Latitude = apartment_metadata['geo']['latitude']
    
                h1_title = soup.find("div", {"id": "units"})
                h1_title = h1_title.find('h1')
                unit_name = h1_title.get_text().strip()
                UnitName = unit_name
    
                unit_id = url[-7:]
                Unit_ID = unit_id
    
                #Apartment Features
                bathroom = main_feature_extractor('Bathroom')
                number_of_bathrooms = int(bathroom[0])
                Bathroom = number_of_bathrooms
    
                Furnished = main_feature_extractor('Furnish')   
    
                area_text = main_feature_extractor('Area')
                area = float(area_text[:-2].strip())
                Area = area
    
                Floor = main_feature_extractor('Floor')
    
                Tower = main_feature_extractor('Tower')
    
                ##Apartment Facilities
                facilities_soup = soup.find_all('span', {"class":"facility-text"})
                facilities = str()
                facilities_list = []
                for _ in facilities_soup:
                    facility = _.get_text().strip()
                    facilities = facilities + facility+' '
                    facilities_list.append(facility)

                AC = facility_checker('AC')
    
                Water_Heater = facility_checker('Water Heater')
    
                Dining_Set = facility_checker('Dining Set')
    
                Electricity = facility_checker('Electricity')
    
                Bed = facility_checker('Bed')
    
                Access_Card = facility_checker('Access Card')
    
                Kitchen = facility_checker('Kitchen')
    
                Fridge = facility_checker('Refrigerator')
    
                Washing_Machine = facility_checker('Washing Machine')
        
                TV = facility_checker('TV')
    
                ATM = facility_checker('ATM')
    
                TV_Cable = facility_checker('TV Kabel')
    
                Grocery = facility_checker('Grocery Store')
    
                Internet = facility_checker('Internet')
    
                Swim_Pool = facility_checker('Kolam Renang')
    
                Laundry = facility_checker('Laundry')
    
                Security = facility_checker('Security')
    
                Basketball = facility_checker('Lapangan Basket')
    
                Multipurpose_Room = facility_checker('Ruang Serbaguna')
    
                Gym = facility_checker('Gym')
    
                Jogging = facility_checker('Jogging Track')
    
                Tennis = facility_checker('Lapangan Tenis')
    
                Restaurant = facility_checker('Restoran')
    
                Playground = facility_checker('Taman Bermain')
    
                Total_Facilities = AC + Water_Heater + Dining_Set + Electricity + Bed + Access_Card + \
                            Kitchen + Fridge + Washing_Machine + TV + ATM + TV_Cable + Grocery + \
                            Internet + Swim_Pool + Laundry + Security + Basketball + Multipurpose_Room + \
                            Gym + Jogging + Tennis + Restaurant + Playground
            
    
                #Apartment Price
                price = soup.find('div', {'class':'price-content'})
                price.find('span',{'class':'text-strikethrough'})
                if price.find('span',{'class':'text-strikethrough'}) is not None:
                    price.find('span',{'class':'text-strikethrough'}).decompose()
                price_raw = price.get_text().replace('\n','').replace(' ','').replace('Rp0','').replace('$0','').replace(',','').replace('$','USD').replace('Rp','IDR').strip()
    
                pattern_year = "['USD|IDR']\d+/['tahun'|'thn']"
                pattern_month = "['USD'|'IDR']\d+/['bulan'|'bln']"
                search_year_regex = re.search(pattern_year, price_raw)
                if search_year_regex is not None:
                    search_year = search_year_regex[0]
    
                if search_year[0] == 'D':
                    currency = 'USD'
                elif search_year[0] == 'R':
                    currency = 'IDR'
                else:
                    currency = 'unknown'
    
                annual_price = search_year[1:].replace('/t','').strip()

                Currency = currency
                Annual_Price = annual_price

                items = [unit_name, Unit_ID, Apt_Name, No_Rooms, Bathroom, Street, Locality, Region, \
                    Longitude, Latitude, Furnished, Area, Floor, Tower, AC, Water_Heater, Dining_Set, Electricity, Bed, \
                    Access_Card, Kitchen, Fridge, Washing_Machine, TV, ATM, TV_Cable, Grocery, Internet, Swim_Pool, Laundry, \
                    Security, Basketball, Multipurpose_Room, Gym, Jogging, Tennis, Restaurant, Playground, Total_Facilities, \
                    Currency, Annual_Price]

                names = ['Unit_Name', 'Unit_ID', 'Apt_Name', 'No_Rooms', 'Bathroom', 'Street', 'Locality', 'Region', \
                    'Longitude', 'Latitude', 'Furnished', 'Area', 'Floor', 'Tower', 'AC', 'Water_Heater', 'Dining_Set', \
                    'Electricity', 'Bed', 'Access_Card', 'Kitchen', 'Fridge', 'Washing_Machine', 'TV', 'ATM', 'TV_Cable', \
                    'Grocery', 'Internet', 'Swim_Pool', 'Laundry', 'Security', 'Basketball', 'Multipurpose_Room', 'Gym', \
                    'Jogging', 'Tennis', 'Restaurant', 'Playground', 'Total_Facilities', 'Currency', 'Annual_Price']

                for i in range(len(items)):
                    st.write(names[i]+': {}'.format(items[i]))
            except:
                st.write("Unknown error occurred. Please insert another apartment unit link and try again.")


        



    elif pages == "Data Visualization":
        st.title('Jabodetabek Apartment Data Visualization')

        if st.checkbox("Display Data", False):
            st.write(dash)

        st.subheader('Unit Type Visualization') 
        ut_chart = st.selectbox("Unit Type Plot", ("Boxplot", "Histogram"))
        if ut_chart == "Boxplot":
            fig, ax = plt.subplots()
            ax = sns.boxplot(x = "Unit Type", y = 'Annual Price', data = dash, order = ['Studio', '1BR', '2BR', '3BR', '4+BR'])
            ax.set(ylabel = "Annual Rent Price (in tens of millions IDR)")
            st.pyplot(fig)
        elif ut_chart == "Histogram":
            fig, ax = plt.subplots()
            ax = sns.countplot(x="Unit Type", data=dash, order = ['Studio', '1BR', '2BR', '3BR', '4+BR'])
            ax.set(ylabel = "Count")
            st.pyplot(fig)

        st.subheader('Area Visualization')
        area_chart = st.selectbox("Area Plot", ("Distribution", "Boxplot", "Scatterplot"))
        if area_chart == "Distribution":
            fig, ax = plt.subplots()
            ax = sns.kdeplot(dash.Area)
            st.pyplot(fig)
        elif area_chart == "Boxplot":
            second_choice = st.selectbox("Categorized By", ("Unit Type", "Region", "Furnished Status"))
            if second_choice == "Unit Type":
                fig, ax = plt.subplots()
                ax = sns.boxplot(x="Unit Type", y="Area", data = dash, order = ['Studio', '1BR', '2BR', '3BR', '4+BR'])
                st.pyplot(fig)
            elif second_choice == "Region":
                fig, ax = plt.subplots()
                ax = sns.boxplot(x="Region", y="Area", data = dash)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            elif second_choice == "Furnished Status":
                fig, ax = plt.subplots()
                ax = sns.boxplot(x="Furnished Status", y="Area", data = dash)
                st.pyplot(fig)
        elif area_chart == "Scatterplot":
            fig, ax = plt.subplots()
            ax = sns.scatterplot(x = "Area", y = 'Annual Price', data = dash, hue = 'Unit Type')
            st.pyplot(fig)
        
        st.subheader('Region Visualization')
        region_chart = st.selectbox("Region Plot", ("Boxplot", "Histogram"))
        if region_chart == "Boxplot":
            fig, ax = plt.subplots()
            ax = sns.boxplot(x = "Region", y = "Annual Price", data = dash)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        elif region_chart == "Histogram":
            fig, ax = plt.subplots()
            ax = sns.countplot(x = "Region", data = dash)
            plt.xticks(rotation = 45)
            st.pyplot(fig)
        
        st.subheader('Jakarta Map Visualization')
        lonlat = dash[['Longitude', 'Latitude', 'Annual Price']].rename({'Longitude':'lon', 'Latitude':'lat'}, axis='columns')
        def get_center_latlong(df):
        # get the center of my map for plotting
            centerlat = (df['lat'].max() + df['lat'].min()) / 2
            centerlon = (df['lon'].max() + df['lon'].min()) / 2
            return centerlat, centerlon
        center_map = get_center_latlong(lonlat)
        
        # create a LinearColorMap and assign colors, vmin, and vmax
        # the colormap will show green for $100,000 homes all the way up to red for $1,500,000 homes
        colormap = cm.LinearColormap(colors=['lightgreen', 'green', 'darkgreen'], vmin=min(lonlat['Annual Price']), vmax=max(lonlat['Annual Price']))

        # create our map again.  This time I am using a different tileset for a new look
        m = folium.Map(location=center_map, zoom_start=10, tiles='OpenStreetMap')
        st.write("The dots represent apartment units from our dataset. The darker the color, the higher the annual rent price.")
        st.write("We can see that most of the darker dots are in Jakarta Selatan region.")

        # Same as before... go through each home in set, make circle, and add to map.
        # This time we add a color using price and the colormap object
        for i in range(len(lonlat.lon)):
            folium.Circle(
                location=[lonlat.iloc[i]['lat'], lonlat.iloc[i]['lon']],
                radius=100,
                fill=True,
                color=colormap(lonlat.iloc[i]['Annual Price']),
                fill_opacity=0.2,
                weight = 5
            ).add_to(m)

        # the following line adds the scale directly to our map
        m.add_child(colormap)
        folium_static(m)


        st.subheader('Facilities Visualization')
        facility = st.selectbox("Select Facility", ('AC', 'Water Heater','Dining Set', 'Electricity', 'Bed', 'Access Card', 
                                                    'Kitchen Set', 'Refrigerator', 'Washing Machine', 'TV', 'ATM', 'TV Cable',
                                                    'Grocery Shop', 'Internet Services', 'Swimming Pool', 'Laundry', 'Security', 
                                                    'Basketball Field', 'Multipurpose Room', 'Gym', 'Jogging Track', 'Tennis Field', 
                                                    'Restaurant', 'Kids Playground'))
        fig, (ax1, ax2) = plt.subplots(ncols = 2)
        sns.boxplot(x=facility, y="Annual Price", data = dash, ax=ax1)
        sns.countplot(x=facility, data=dash, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("On the X-Axis, '0' represents units without the selected facility, and '1' represents units with the selected facility")
    elif pages == "Play with Machine Learning Models":
        st.title('Train and evaluate your own Machine Learning models')
        st.subheader('Do you need some introduction to machine learning models, and what this page should do?')
        explanation = st.radio('Choose', ("Yes, provide me with some explanation, please.", "No, I'm familiar with the subject matter and would like to \
                train the model right away."), index=1)
        if explanation == "Yes, provide me with some explanation, please.":
            st.subheader('Part 1: What is a Machine Learning Regression model?')
            st.write("In this project, we use data which contain apartment unit details (location, area, facilities, etc) to predict \
                its annual rent price. The data about our units are called 'predictors' or 'independent variables', while the \
                    annual rent price is called the 'target' or 'dependent' variable.")
            st.write("Simply put, a machine learning regression algorithm trains a model to be able to predict the 'target variable' as \
                'accurate' as possible by learning from existing data (which has predictors with their matching target variables'). ")
            st.write("Our model learns from the data we collect and try to come up with a set of rule / calculation of its own, so when \
                we enter a new unit's details, it can give a good prediction, on what its rent price might be.")
            st.subheader("Part 2: How to measure a model's performance?")
            st.write("We have over 5000 rows of data, and we split it randomly into two sets: \
                the training set (usually around 80-90 percent of all data), and the testing set. Our model is trained on the training \
                set, and we'll ask them to guess the annual rent price of the testing set without providing it with the 'target' variable. \
                Since we have the 'correct' answer, we can compare the model's prediction (its guess of rent price) with the actual rent price \
                of those units.")
            st.write("A good model should perform well however we split the data. In scikit-learn, the Python framework used for Machine Learning \
                in this project, the 'randomness' of the data splitting is determined by a 'random seed'. Any 'random seed' should do the job just fine, \
                but if you want to compare multiple models performance, it's best to use the same 'random seed' all across your models. This means that \
                each model is trained on the same training set and tested on the same testing set.")
            st.subheader('Part 3: What metrics can be used to measure model performance?')
            st.write("Our model performance is scored by a few metrics. MAE is an abbrv. of 'Mean Absolute Error', \
                showing how much our model's prediction differ from the actual data. RMSE is an abbrv of 'Root Mean Squared Error', in which  \
                differences between our prediction and the actual data is squared, averaged out and then square rooted. This metric 'punishes' \
                huge errors more than the first one.")
            st.write("In a simpler term, 'if you think having an error of 4 unit is twice as bad as having \
                an error of 2, use the MAE. If you think having an error of 4 is more than twice as bad as having an error of 2, because you want to \
                punish larger mistakes, use the RMSE.' We aim for our RMSE and MAE to be as low as possible, but since we're talking about \
                apartment rent prices in the millions, it made sense for our RMSE and MAE to be in the millions too.")
            st.write('The final metric, R2 (R-squared), is on a scale of 0 to 1 - and in a simple term, explains how much of variances (you coudl say movements) \
                    in our actual data can be explained by our model. The higher the score, the better our model is at predicting the actual relationship \
                    between the features and the target variable(s). This is one of the metrics that I really like - as regardless the value of our target variable, \
                    and R-squared score always have a range from 0 to 1.')
            st.write('For example, a model predicting the price of fruits at the supermarket will naturally tend to have lower RMSE and MAE scores, as the \
                    target variable is in thousands of rupiah, not millions. However, the R-squared score always ranges from 0 to 1.')
            st.subheader("Part 4: The Student Analogy - Underfitting and Overfitting")
            st.write('A machine learning model is like a student who wishes to learn some materials for an upcoming exam. The student has to \
                figure out how to understand the available materials, so he/she could give an accurate answer when given a new question. \
                The study materials are our trainng data, and the set of new questions on the test is the test data.')
            st.write('To really test if a student understand the lesson, we need to look at his/her performance on the test, as it is something \
                our student has not encountered yet. If our student gets a good grade on the test, it means he/she understands the material well enough.')
            st.write("What does this analogy has to do with underfitting and overfitting? Suppose we measure our student's performance twice. First, by asking \
                him/her to answer exercise questions in the book (predicting the train set), and answering completely new questions on the test (predicting test data).")
            st.write("If our student has a lower score on the exercise than the real test (having a better accuracy on data it has never seen) - it means \
                that our student is lazy. He could've performed better. This is called underfitting. An underfit model is a model that's too general, and does not \
                study well enough on the train set.")
            st.write("If our student has a high score in answering exercise questions, but gets a low test score, it means our student does not really understand the materials - \
                he/she just memorizes the exercise questions answer key. That's why our student performed very well on the exercise questions. However, \
                when presented with new unseen questions, he/she fails to answer correctly. This is called overfitting.")
            st.write("When overfitting happens, our model only works well on 'training' set, but does not have a good score on the test set. It means, \
                if it is given new data, it cannot predict well enough")
            st.write("We want a model with good accuracy, but does not underfit nor overfit. This is the delicate part of tuning our model.")
            st.subheader('Epilogue')
            st.write('Last but not least, although having an interactive dashboard like this may make it seem easy to train and evaluate machine \
                learning models, it has some limitations. It is much better to train and evaluate model using Jupyter notebooks/Google colabs as \
                we can do more things with lines of codes rather than a point-and-click interface. Furthermore, complex models can be \
                trained faster by the use of GPU (Graphics Processing Unit) which is very difficult to implement in web apps like these.')                    
        
        st.subheader('What score should I aim for?')    
        st.write("When cross validated, the final model's RMSE score usually averages at around 30 000 000, and has an R-squared score around 0.9. \
            Try to see if you can find combinations of columns and parameters that yields a model with RMSE score around the final model's, \
            but watch out for overfitting!")


        st.subheader('Which features would you like to include?')
        cols = st.multiselect("Choose your feature columns", ('Number of Bedrooms', 'Bathroom', 'District', 'Region',
                                'Longitude', 'Latitude', 'Furnished', 'Area', 'AC', 'Water Heater','Dining Set', 'In-Jakarta Check',
                                'Electricity', 'Bed', 'Access Card', 'Kitchen Set','Refrigerator', 'Washing Machine', 
                                'TV', 'ATM', 'TV Cable','Grocery Shop', 'Internet Services', 'Swimming Pool', 'Laundry',
                                'Security', 'Basketball Field', 'Multipurpose Room', 'Gym', 'Jogging Track', 'Tennis Field', 
                                'Restaurant', 'Kids Playground', 'Total Number of Facilities'))

        st.subheader('Select a random seed')
        st.write('Any number of seed will do just fine, but if you wish to compare multiple models consecutively, select the same random seed \
            so those models will have the same train and test set.')
        seed = st.slider('Seed', 0, 10000)
        st.subheader('Select the test data proportion')
        st.write('This represents how much rows will be taken as the test set.')
        test_size = st.slider('Test Proportion', 0.1, 0.3)

        X_Custom = ML_Ready[cols]
        y = ML_Ready['Annual Price']

        labelencoder=LabelEncoder()
        for col in X_Custom.columns:
            if col == 'Region':
                X_Custom[col] = labelencoder.fit_transform(X_Custom[col])
            elif col == 'District':
                X_Custom[col] = labelencoder.fit_transform(X_Custom[col])
            
        X_train, X_test, y_train, y_test = train_test_split(X_Custom, y, test_size = test_size, random_state = seed)
        test_val = y_test.to_numpy()
        train_val = y_train.to_numpy()

        
        #function for giving prediction results
        def predict_model(model, logtr = False):
            if logtr == False:
                predict_test = model.predict(X_test)
                predict_train = model.predict(X_train)
            else:
                predict_test = np.expm1(model.predict(X_test))
                predict_train = np.expm1(model.predict(X_train))

            return predict_test, predict_train

        #function to draw scatterplots of predicted vs actual train/test values
        def plot_result():
            fig, (ax1, ax2) = plt.subplots(ncols = 2)  #scatterplot for both train and test values
            sns.scatterplot(test_val, predict_test, ax=ax1)
            sns.scatterplot(train_val, predict_train, ax=ax2)
            ax1.set(xlabel = "Actual Test Price Values")
            ax1.set(ylabel = "Predicted Test Price Values")
            ax2.set(xlabel = "Actual Train Price Values")
            ax2.set(ylabel = "Predicted Train Price Values")
            plt.tight_layout()
            st.pyplot(fig)

        def evaluate():
            from sklearn import metrics
            RMSE_test = np.sqrt(metrics.mean_squared_error(test_val, predict_test))
            R2_test = metrics.r2_score(test_val, predict_test)
            RMSE_train = np.sqrt(metrics.mean_squared_error(train_val, predict_train))
            R2_train = metrics.r2_score(train_val, predict_train)

            st.write('RMSE of Test Set Prediction:', RMSE_test)
            st.write('R2 of Test Set Prediction:', R2_test)
            st.write('RMSE of Train Set Prediction:', RMSE_train)
            st.write('R2 of Train Set Prediction:', R2_train)

            if R2_train < R2_test:
                    st.write('Your R-squared train score is less than R-squared test score. Your model might be underfit. \
                            Try to change your feature columns to see if you can have a better fit.')
            elif R2_train > R2_test:
                if (R2_train - R2_test)*100 > 5.9:
                    st.write('Your R-squared train score is higher than your R-squared test score by 6 percent or more. \
                        Your model might be overfit. Try to remove one or two feature columns, do a different range of \
                        hyperparameter tuning, or choose a different combination of columns')
                else:
                    if R2_test < 0.75:
                        st.write('Your model does not underfit nor overfit, but its R-squared score is lower than 0.75.')
                        st.write("You can still do better! Try to find another combination of columns and/or parameters. \
                            A linear regression can achieve up to 0.78 R-squared score, \
                            while XGBoost and Light GBM can reach up to 0.9 R-squared score.")
                    elif R2_test < 0.88:
                        st.write("Congratulations! You have made a regression model that is not underfit nor overfit, and with a relatively good \
                            R-squared score. You haven't beat the model deployed on 'Calculator' page, though :)")
                    else:
                        st.write("Congratulations! You have made a regression model that is not underfit nor overfit, and your model's performance \
                            is as good as my final model. Well done!")

                
            


        st.subheader('Specify your machine learning model')
        modeltype = st.selectbox('Model Type', ('Linear Regression', 'XGBoost', 'Light GBM Regressor'))
        if modeltype == "Linear Regression":
            if st.button("Fit and Evaluate Model"):
                
                lm = LinearRegression()
                lm.fit(X_train, y_train)
                predict_test, predict_train = predict_model(lm)

                plot_result()

                evaluate()
                                
        
        elif modeltype == 'XGBoost':
            if explanation == "Yes, provide me with some explanation, please.":
                st.write("What is hyperparameter tuning? A hyperparameter is a parameter whose value is used to control the learning process \
                    of our machine learning model. As an analogy, picture a machine learning model as a student who needs to study the 'training' \
                    set in order to prepare for the exam, which is the 'testing' set. Hyperparameters are things that affect how this student learns.") 
                st.write("For example, how much he/she learns on a single day? Or how many days before the exam that he/she studies? Regardless of these \
                    'hyperparameters' our student still learns the same data - just in slightly different ways. Finding the optimal 'hyperparameter' \
                    is like finding our student's best study routine - so he can achieve better result during the exam.")

            st.subheader('Do you want to do hyperparameter tuning?')
            tune_check = st.radio('Answer', ('Yes', 'No, use the baseline model.'))
            if tune_check == "Yes":
                st.markdown('Note: In this simulation, we will only be doing tuning on four parameters. Refer to the [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/parameter.html) \
                    to see the list of all parameters and their definition.')
                new_XGB = XGBRegressor()
                st.subheader("Search range for 'min_child_weight' parameter:")
                min_ch_w_range = st.slider('Select range', 1, 50, (3, 20))
                st.subheader("Search range for 'max_depth' parameter:")
                max_depth_range = st.slider('Select range', 1, 50, (2, 20))
                st.subheader("Search range for 'subsample' parameter:")
                subsample_range = st.slider('Select range', 0.1, 1.0, (0.2, 0.8))
                st.subheader("Search range for 'colsample_bytree' parameter:")
                colsample_range = st.slider('Select range', 0.1, 1.0, (0.3, 0.8))

                params = {
                    'min_child_weight': min_ch_w_range,
                    'subsample': subsample_range,
                    'colsample_bytree': colsample_range,
                    'max_depth': max_depth_range
                    }

                skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
                random_search = RandomizedSearchCV(new_XGB, param_distributions=params, n_iter=50, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1, cv=skf.split(X_Custom,y), verbose=2, 
                                   random_state=seed)
                st.subheader('Do you want to perform log transformation on the target variable?')
                st.markdown('If you choose yes, then the trained target variable will be log-transformed first. Then, the prediction \
                            result from X_test will be exponentially transformed to match the original scale of Annual Price - before \
                            comparing it against the test actual data.')
                log_trf = st.radio("Log Transformation", ("Yes", "No"))
                if st.button("Find best parameter and evaluate the tuned model"):
                    with st.spinner('Searching for best parameter(s)...'):
                        random_search.fit(X_Custom, y)
                        st.success('Done!')

                    min_child_weight = random_search.best_params_['min_child_weight']
                    max_depth = random_search.best_params_['max_depth']
                    subsample = random_search.best_params_['subsample']
                    colsample_bytree = random_search.best_params_['colsample_bytree']
                    
                    st.write('min_child_weight:', min_child_weight)
                    st.write('max_depth:', max_depth)
                    st.write('subsample:', subsample)
                    st.write('colsample_bytree:', colsample_bytree)
                    tuned_newxgb = XGBRegressor(min_child_weight = min_child_weight,
                                                max_depth = max_depth,
                                                subsample = subsample,
                                                colsample_bytree = colsample_bytree)

                    if log_trf == "No":
                        tuned_newxgb.fit(X_train, y_train)
                        predict_test, predict_train = predict_model(tuned_newxgb)
                        plot_result()
                        evaluate()

                    else:
                        tuned_newxgb.fit(X_train, np.log1p(y_train))
                        predict_test, predict_train = predict_model(tuned_newxgb, logtr=True)
                        plot_result()
                        evaluate()
                                    
            else:
                st.subheader('Do you want to perform log transformation on the target variable?')
                st.markdown('If you choose yes, then the trained target variable will be log-transformed first. Then, the prediction \
                            result from X_test will be exponentially transformed to match the original scale of Annual Price - before \
                            comparing it against the test actual data.')
                log_trf = st.radio("Log Transformation", ("Yes", "No"))
                if st.button("Train and evaluate the model"):
                    if log_trf == "No":
                        standard_xgb = XGBRegressor()
                        standard_xgb.fit(X_train, y_train)
                        
                        predict_test, predict_train = predict_model(standard_xgb)
                        plot_result()
                        evaluate()
                
                    if log_trf == "Yes":
                        standard_xgb = XGBRegressor()
                        standard_xgb.fit(X_train, np.log1p(y_train))
                        
                        predict_test, predict_train = predict_model(standard_xgb, logtr=True)
                        plot_result()
                        evaluate()

        elif modeltype == 'Light GBM Regressor':
            if explanation == "Yes, provide me with some explanation, please.":
                st.write("What is hyperparameter tuning? A hyperparameter is a parameter whose value is used to control the learning process \
                    of our machine learning model. As an analogy, picture a machine learning model as a student who needs to study the 'training' \
                    set in order to prepare for the exam, which is the 'testing' set. Hyperparameters are things that affect how this student learns.") 
                st.write("For example, how much he/she learns on a single day? Or how many days before the exam that he/she studies? Regardless of these \
                    'hyperparameters' our student still learns the same data - just in slightly different ways. Finding the optimal 'hyperparameter' \
                    is like finding our student's best study routine - so he can achieve better result during the exam.")

            st.subheader('Do you want to do hyperparameter tuning?')
            tune_check = st.radio('Answer', ('Yes', 'No, use the baseline model.'))
            if tune_check == "Yes":
                st.markdown("Note: In this simulation, we will only be doing tuning on four parameters. Refer to the [Light GBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) \
                    to see the list of all parameters and their definition.")
                new_LGB = LGBMRegressor()
                st.subheader("Search range for 'num_leaves' parameter:")
                num_leaves = st.slider('Select range', 1, 50, (20, 40))
                st.subheader("Search range for 'min_data_in_leaf' parameter:")
                min_data_in_leaf = st.slider('Select range', 1, 30, (2, 15))
                st.subheader("Search range for 'learning_rate' parameter:")
                learning_rate = st.slider('Select range', 0.1, 1.0, (0.2, 0.8))
                st.subheader("Search range for 'max_bin' parameter:")
                max_bin = st.slider('Select range', 100, 300, (200, 260))

                params = {
                    'num_leaves': num_leaves,
                    'min_data_in_leaf': min_data_in_leaf,
                    'learning_rate': learning_rate,
                    'max_bin': max_bin
                    }

                skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
                random_search = RandomizedSearchCV(new_LGB, param_distributions=params, n_iter=50, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1, cv=skf.split(X_Custom,y), verbose=2, 
                                   random_state=seed)
                st.subheader('Do you want to perform log transformation on the target variable?')
                st.markdown('If you choose yes, then the trained target variable will be log-transformed first. Then, the prediction \
                            result from X_test will be exponentially transformed to match the original scale of Annual Price - before \
                            comparing it against the test actual data.')
                log_trf = st.radio("Log Transformation", ("Yes", "No"))
                if st.button("Find best parameter and evaluate the tuned model"):
                    with st.spinner('Searching for best parameter(s)...'):
                        random_search.fit(X_Custom, y)
                        st.success('Done!')

                    num_leaves = random_search.best_params_['num_leaves']
                    min_data_in_leaf = random_search.best_params_['min_data_in_leaf']
                    learning_rate = random_search.best_params_['learning_rate']
                    max_bin = random_search.best_params_['max_bin']
                    
                    st.write('num_leaves:', num_leaves)
                    st.write('min_data_in_leaf:', min_data_in_leaf)
                    st.write('learning_rate:', learning_rate)
                    st.write('max_bin:', max_bin)
                    tuned_newlgb = LGBMRegressor(num_leaves = num_leaves,
                                                min_data_in_leaf = min_data_in_leaf,
                                                learning_rate = learning_rate,
                                                max_bin = max_bin)

                    if log_trf == "No":
                        tuned_newlgb.fit(X_train, y_train)
                        predict_test, predict_train = predict_model(tuned_newlgb)
                        plot_result()
                        evaluate()

                    else:
                        tuned_newlgb.fit(X_train, np.log1p(y_train))
                        predict_test, predict_train = predict_model(tuned_newlgb, logtr=True)
                        plot_result()
                        evaluate()

            
            else:
                st.subheader('Do you want to perform log transformation on the target variable?')
                st.markdown('If you choose yes, then the trained target variable will be log-transformed first. Then, the prediction \
                            result from X_test will be exponentially transformed to match the original scale of Annual Price - before \
                            comparing it against the test actual data.')
                log_trf = st.radio("Log Transformation", ("Yes", "No"))
                if st.button("Train and evaluate the model"):
                    if log_trf == "No":
                        standard_lgb = LGBMRegressor()
                        standard_lgb.fit(X_train, y_train)
                        predict_test, predict_train = predict_model(standard_lgb)
                        plot_result()
                        evaluate()
                
                    if log_trf == "Yes":
                        standard_lgb = LGBMRegressor()
                        standard_lgb.fit(X_train, y_train)
                        predict_test, predict_train = predict_model(standard_lgb, logtr=True)
                        plot_result()
                        evaluate()
                               
    
    
    elif pages == "Summary":
        st.title ('Summary')

        st.subheader('Dataset Description')
        st.markdown('The data consists of 5340 apartment units scraped from Jendela 360 website on December 3rd 2020.')
        st.markdown('There are 6 numerical features: number of bedrooms, number of bathrooms, longitude, latitude, area, and total number of facilities.')
        st.markdown("Other than these numerical features, there are 23 categorical features which are labeled '1' if the facility is \
            present in the apartment, or '0' if the facility is not present.")
        st.markdown("The area of apartments in the dataset ranges from 20 - 250 meter squared.")
        st.markdown("The price of the apartments in the dataset ranges from 12 - 705.6 million rupiah")

        st.subheader('Regression Model Used')

        models = {'Model Name': ['Linear Regression', 'KNN Regressor', 'Random Forest Regressor', 'XGB Regressor (baseline)', 'XGB Regressor (tuned)', 'Light GBM Regressor (baseline)', 'Light GBM Regressor (tuned)'],
            'Rounded R2 Score*': [0.8, 0.84, 0.9, 0.9, 0.9, 0.9, 0.9],
            'Further Description': ['Suffers from multicolinearity issues', 'Data has to be scaled', 'Hyperparameter tuning does not improve the model', 'Overfits', 'Accuracy slightly below baseline Light GBM', 'Model of choice', 'Overfits']       
        }
        
        model_report = pd.DataFrame(models, columns = ['Model Name', 'Rounded R2 Score*', 'Further Description'])
        st.write(model_report)

        st.write('*: Exact R squared score depend on how we split the training and testing data. Overall, the Random Forest, XGB, and Light GBM Regressors \
            have similar R squared score, but we provide the reasons on why ultimately we choose the baseline Light GBM Model as our final model.')
        st.markdown('XGB vs Light GBM: it seems that in this dataset, the baseline XGB model tends to have higher accuracy on test set, but it overfits on the training set - while \
            the baseline Light GBM model does not overfit on the training set. The opposite happens when we tune the model. After hyperparameter tuning, \
            the XGB model does not overfit, but the Light GBM becomes overfit.')
        st.markdown('Based on the bias-variance tradeoff, ultimately the author chooses a model that does not overfit. The baseline XGB model, although \
            performing slightly better (by 0.5-1 percent) than the baseline LGB model, achieves a 0.98 R2-score on the training set, and is thus not chosen due to this overfitting issue. \
            By comparison, the baseline LGB model has an R2-score of 0.95 on the training set, and 0.9 on testing set.')
        st.markdown("Other than single prediction value, we also generate a '99%' and '95%' prediction interval from our Light GBM model.")

        st.subheader('Rank of Feature based on Importance')
        st.markdown('Feature importance is ranked based on how many of these features are used to make splits in the decision trees (in the Light GBM algorithm)')
        feature_imp = pd.read_csv('Temp Files\Feature Importance.csv')
        st.write(feature_imp)

        st.subheader('Insights')
        st.write('- The most important features are area, location (latitude and longitude), total number of facilities, and number of (bed)rooms.')
        st.write('- There are 3 sport facilities which are ranked higher in the feature importance ranking: tennis field, basketball field, and jogging track.')
        st.write('- There are 2 sport facilities which are ranked lower in the feature importance ranking: swimming pool and gym.')
        st.write('- Most apartments in Jakarta, regardless of their price/luxury, have swimming pool and gym. This is why the presence of both \
            facilities does not impact apartment annual rent price a lot. However, swimming pools and gym in expensive apartments are almost always better, larger, and \
            having more expensive equipments. Unfortunately, our dataset only measures the presence of such facilities, and does not measure, for example, the area of swimming pool, \
            the quality of gym sets, etc. The author speculate that if we can have data on the details of these facilities, perhaps they can be \
            more relevant in determining the annual rent price of our apartment.')
        st.write('- The most important in-room facilities are washing machine, fridge, and kitchen. Perhaps, lower priced apartments, even though are listed as fully furnished, \
            does not provide these facilities. Lower priced apartments perhaps rely on public laundry as a means to wash their daily wears, instead of private washing machines in each room.')
        
        st.subheader('Future Improvement Suggestions')
        st.write('Although our model reached a desirable R-squared score, it is by no means perfect. Here are a few suggestions for future improvements:')
        st.write('- Gather data from other online apartment rental/marketplace. Our dataset only comes from a single website, which can make it quite skewed if \
            the Jendela360 website is not representative of the whole apartment market in Jakarta.')
        st.write("- Add more features regarding nearby places to our apartments. For example, how many schools, MRT stations, shopping malls, hospitals are there within X radius of our apartment. \
            Usually, apartment which are placed 'strategically' (near public places) are priced higher.")
        st.write("- Perhaps, knowing who the developers of our apartments might also give us a clue on the apartment annual rent price. Certain developers \
            tend to develop higher priced apartments.")

    elif pages == 'FAQs':
        st.title('FAQs')        

        st.subheader("Q: Tell us again why you choose this topic?")
        st.write("A: As a young adult, renting apartment units seem to be the preference for having a living space, especially \
            in the short term time frame. Buying properties seem to be too expensive for us in the beginning of our careers. \
            However, when I am searching for apartment units online, I'm curious on how to know if an advertised unit is overpriced or not.")
        st.write("I feel that there needs to be a mathematical equation or model to make sense of apartment unit prices. Which factors \
            determine an apartment unit's rental price the most? Is it area? Location? Facilities? Has there been a study about this yet?")
        st.write("I think that a simple web app that can help estimate the price of an apartment unit can be beneficial for those who are seeking \
            places to live, or those who are planning to rent their apartment for extra income - so they can settle on a price that's standardized \
            and not over or underpriced.")
        st.subheader("Q: How do you get the data?")
        st.write("A: I scraped the data from a third party website named 'Jendela 360'. It's a website where we can view apartment units \
            online and see its details before contacting the marketing agency to rent it - which also can be done online. In simpler terms, \
            it's an online marketplace for property renting/buying/selling.")
        st.write("I scrape the apartment rental section of it using \
            Beautiful Soup and Selenium libraries in Python. I scraped it on the third week of October 2020, so if somehow you attempt to \
            scrape it again now, it might not yield the exact same result - but our data should be representative enough for the current apartment rent market.")
        st.subheader("Q: After you get the data, what do you do?")
        st.write("A: I'm glad you asked. Here's the thing - I spend more time cleaning the data than training it. As the data is raw, I need to \
            check for values which does not make sense. If you check my first jupyter notebook in my 'jakpartment' repository, you'll find that \
            there are a few studio apartment units which are priced outstandingly high because they attach a wrong currency to it.")
        st.write("There's a studio apartment unit priced at 30 million...wait for it...USD. There are also clearly furnished apartment units which \
            does not list any facilities. There are also an apartment unit which has over 7000 meter squared of area. Upon further checking, that \
            apartment unit is a condominium penthouse - which is not what we're looking for.")
        st.write("Before I train my data, I need to remove outliers, fix wrongly labeled values (in terms of currency unit), and visualize the data. \
            After I'm sure that the data is cleaned, I proceed to train it.")
        st.subheader("Q: Wow. That sounds like a handful. Do you think this is often overlooked?")
        st.write("A: Yes! Many people think that machine learning or data science in general is about creating fancy models with high accuracy, \
            but remember - 'Garbage in, garbage out'. You can't have a good model without a good data. Training is arguably less of a hassle as \
            it can be automated, but cleaning data needs some kind of an 'instinct' to do, because there are no fixed rules on how to do it. \
            Different raw data needs different cleaning and wrangling treatments before they are ready to be processed.")
        st.subheader("Q: So, the training process is easy, right?")
        st.write("A: Not really. There are complexities to the training process too. We have to choose which algorithm works best for our data. \
            As my data here has a lot of multicolinearity issues, and are not scaled the same (longitude is in the hundreds, latitude is a negative value, and \
            area is highly skewed), linear regression wouldn't work as well. This might also be true as the target variable (Annual Price) is not entirely linear too. \
            Understanding my data structure and shortcomings make me decide to use Light GBM and XGB as my final two models.")
        st.write("Furthermore, we must also consider the dangers of underfitting and overfitting. Baseline model of XGBoost Regressor gives me a high \
            test set score, but it performed even higher on the train test score. It reaches 0.98 R-squared score when it predicts our training set. This is \
            a clear indication of overfitting. Ultimately, this is what caused me to choose the Light GBM model.")
        st.subheader("Q: Last but not least, what do you personally learn while doing this project?")
        st.write("A: This project really gets my hand dirty with data. An end-to-end project like this allows me to explore most aspects \
            of a Data Science project workflow, from gathering the data to deploying the model.")   
    elif pages == "About the Author":
        st.title('About the Author')
        st.subheader("Greetings! My name is Grady Matthias Oktavian. Nice to meet you! And thank you for spending time in my first end-to-end Data Science project.")
        st.write("I graduated at 2020 from Universitas Pelita Harapan with the title Bachelor of Mathematics, majoring in Actuarial Science. \
            Currently, I'm studying at Pradita University as a student in the Master of Information Technology degree majoring in Big Data and IoT. \
            I am also employed as an IFRS 17 Actuary at PT Asuransi Adira Dinamika.") 
        st.write("I like learning about statistics and mathematics. Since today is the age of Big Data, I find that most people who \
            aren't majoring in mathematics might find themselves overwhelmed with large amounts of data. My dream is to help people make better \
            decision through a data-driven approach.") 
        st.write("In order to do that, I am happy to wrangle and analyze raw data, creating models based on it, and \
            conveying insights I gained to others who are not well-versed in data, so they can understand it without having to \
            get their hands dirty with the data. I hope through my help, people can understand things better, busineess owners can \
            make better contingency plans and take better decisions.")
        st.markdown("Currently, I am certified by Google as a [TensorFlow Developer](https://www.credential.net/794f2bb6-d377-4b5b-ac9d-9d3bed582d2d), \
            and a [Cloud Professional Data Engineer](https://www.credential.net/df7c3d9d-011a-41fd-9d64-49ada5a0619c#gs.ktrahi).")
        st.write("If you wish to have a suggestion for this project, or contact me for further corresnpondences, \
            please reach out to me via email at gradyoktavian@gmail.com, or send me a message to \
            my [LinkedIn profile](https://www.linkedin.com/in/gradyoktavian/).")
        st.write("Thank you! Have a nice day.")


    
    
if __name__ == '__main__':
    main()


