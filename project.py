import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st

st.title("Rental House Prices Prediction")

# Define dropdown options
house_type_list = [
    "3 BHK Independent Floor",
    "4 BHK Independent Floor",
    "2 BHK Independent Floor",
    "3 BHK Apartment",
    "4 BHK Villa",
    "2 BHK Apartment",
    "1 BHK Independent Floor",
    "1 BHK Apartment",
    "5 BHK Villa",
    "5 BHK Independent Floor",
    "1 RK Studio Apartment",
    "5 BHK Independent House",
    "4 BHK Independent House",
    "4 BHK Apartment",
    "3 BHK Independent House",
    "2 BHK Independent House",
    "6 BHK Independent Floor",
    "1 BHK Independent House",
    "5 BHK Apartment",
    "9 BHK Independent House",
    "8 BHK Independent Floor",
    "7 BHK Independent Floor",
    "10 BHK Independent House",
    "6 BHK penthouse",
    "8 BHK Independent House",
    "8 BHK Villa",
    "7 BHK Independent House",
    "12 BHK Independent House"
]


locations = [
    'Kalkaji', 'Mansarover Garden', 'Uttam Nagar', 'Model Town', 'Sector 13 Rohini', 'DLF Farms', 'Laxmi Nagar', 
    'Swasthya Vihar', 'Janakpuri', 'Pitampura', 'Gagan Vihar', 'Dabri', 'Govindpuri Extension', 'Paschim Vihar', 
    'Vijay Nagar', 'Vasant Kunj', 'Safdarjung Enclave', 'Hauz Khas', 'Bali Nagar', 'Rajouri Garden', 
    'Shalimar Bagh', 'Green Park', 'Dr Mukherji Nagar', 'Subhash Nagar', 'DLF Phase 5', 'Patel Nagar', 'Jasola', 
    'Dwarka Mor', 'Kaushambi', 'Surajmal Vihar', 'Sector 4 Dwarka', 'Sector 6 Dwarka', 'Sector 14 Dwarka', 
    'Sarvodaya Enclave', 'Chattarpur', 'Ramesh Nagar', 'Mayur Vihar II', 'Naraina', 'Greater Kailash', 
    'Chittaranjan Park', 'Sector 19 Dwarka', 'Sector 23 Dwarka', 'Lajpat Nagar III', 'South Extension 2', 
    'Sector-18 Dwarka', 'Mansa Ram Park', 'Gautam Nagar', 'Sector 22 Dwarka', 'Sheikh Sarai', 'Govindpuri', 
    'Sector 13 Dwarka', 'Shanti Niketan', 'Defence Colony', 'Malviya Nagar', 'Sector 23 Rohini', 'Kirti Nagar', 
    'Badarpur', 'Lajpat Nagar I', 'Sector-7 Rohini', 'Sector 23B Dwarka', 'Vikaspuri', 'Sultanpur', 
    'Sector 11 Dwarka', 'Karampura', 'Munirka', 'Mahavir Enclave', 'Greater Kailash 1', 'Panchsheel Park', 
    'Sector 12 Dwarka', 'Sector 7 Dwarka', 'Bindapur', 'Alaknanda', 'Sitapuri', 'Dashrath Puri', 'Manglapuri', 
    'Sector 8 Dwarka', 'Sector 5 Dwarka', 'Kalyan Vihar', 'Sector-B Vasant Kunj', 'Green Park Extension', 
    'Safdarjung Development Area', 'Panchsheel Enclave', 'Lajpat Nagar', 'Shastri Nagar', 'Jor Bagh', 
    'Golf Links', 'Vasant Vihar', 'Anand Niketan', 'Anand Lok', 'East of Kailash', 'Gulmohar Park', 
    'Zone L Dwarka', 'Raja Garden', 'Kalu Sarai', 'Tagore Garden Extension', 'Saket', 'Sector 2 Dwarka', 
    'Geeta Colony', 'Anand Vihar', 'Ashok Nagar', 'Dilshad Garden', 'Gujranwala Town', 'Sector 10 Dwarka', 
    'Sector 16 Dwarka', 'Palam', 'Vikas Puri', 'Masjid Moth Village', 'Sewak Park', 'Sagar Pur', 'Kamla Nagar', 
    'Ajmeri Gate', 'Rajpur', 'Jangpura', 'Greater Kailash II', 'Garhi', 'Nizamuddin East', 'Ansari Nagar West', 
    'Sat Bari', 'Central Ridge Reserve Forest', 'New Friends Colony', 'Sector 3 Dwarka', 'Sector 9 Dwarka', 
    'Moti Bagh', 'Sainik Farm', 'Karol Bagh', 'Sarvpriya Vihar', 'Uday Park', 'Kailash Hills', 
    'Geetanjali Enclave', 'Soami Nagar', 'Masoodpur', 'Mehrauli', 'Shakurpur', 'Razapur Khurd', 'Matiala', 
    'Khirki Extension', 'Sector 11 Rohini', 'Sector 8', 'Khushi Ram Park Delhi', 'Dwarka Sector 17', 
    'Preet Vihar', 'Mayur Vihar Phase 1', 'Rajpur Khurd Village', 'Freedom Fighters Enclave', 'Inderpuri', 
    'Rajpur Khurd Extension', 'Navjeevan Vihar', 'Vishnu Garden', 'Shahdara', 'Patparganj', 'IP Extension', 
    'Punjabi Bagh', 'AGCR Enclave', 'Rajinder Nagar', 'Krishna Nagar', 'Niti Bagh', 'Shakurbasti', 
    'Sundar Nagar', 'Sector 11', 'Sector 16A Dwarka', 'Guru Angad Nagar', 'Sector 7 Dwarka New Delhi', 
    'Tuglak Road', 'Maharani Bagh', 'Friends Colony', 'Moti Nagar', 'New Moti Nagar', 'Shivalik', 
    'Shahpur Jat Village', 'Naraina Vihar', 'Sector 1 Dwarka', 'Tihar Village', 'Nizamuddin West', 'Ladosarai', 
    'Haiderpur', 'New Ashok Nagar', 'Jangpura Extension', 'Neb Sarai', 'Sunder Nagar', 'Mayur Vihar Phase II', 
    'West End', 'Ghitorni', 'Prithviraj Road', 'Malcha Marg', 'Lodhi Road', 'Tilak Marg', 
    'B1 Block Paschim Vihar', 'Sector 6 Rohini', 'New Rajinder Nagar', 'Aurungzeb Road', 'Amrita Shergill Marg', 
    'Babar Road', 'Lodhi Gardens', 'Lodhi Estate', 'East Patel Nagar', 'Sector 17 Dwarka', 'B 5 Block', 
    'New Rajendra Nagar', 'Lajpat Nagar IV', 'Prakash Mohalla Amritpuri', 'Rohini Sector 9', 'Old Rajender Nagar', 
    'Mayur Vihar 2 Phase', 'Dwarka 11 Sector', 'Dwarka Sector 12', 'Kishan Ganj', 'IP Extension Patparganj', 
    'Sector 14 Rohini', 'Amritpuri', 'Jamia Nagar', 'Kailash Colony', 'Prakash Mohalla', 'Hemkunt Colony', 
    'Chhatarpur Extension', 'Lajpat Nagar II', 'Connaught Place', 'Uttam Nagar West', 'Poorvi Pitampura', 
    'Vaishali Dakshini Pitampura', 'Uttari Pitampura', 'Sector 9 Rohini', 'Vasant Kunj Sector A', 
    'Sector B Vasant Kunj', 'Baljeet Nagar', 'Panchsheel Vihar', 'Lok Vihar', 'Dakshini Pitampura', 
    'Kohat Enclave', 'Saraswati Vihar', 'Prashant Vihar Sector 14', 'Engineers Enclave Harsh Vihar', 
    'Tarun Enclave', 'Block MP Poorvi Pitampura', 'Block PP Poorvi Pitampura', 'Kapil Vihar', 
    'Hauz Khas Enclave', 'Westend DLF Chattarpur Farms', 'Abul Fazal Enclave Jamia Nagar', 'Fateh Nagar', 
    'Pitampura Vaishali', 'Block DP Poorvi Pitampura', 'Block AP Poorvi Pitampura', 'Block WP Poorvi Pitampura', 
    'Sector 28 Rohini', 'Rohini Sector 16', 'Block A3', 'Uttam Nagar East', 'Mahipalpur', 'Hari Nagar', 
    'Tri Nagar', 'Jhil Mil Colony', 'Yojna Vihar', 'Khanpur', 'West Patel Nagar', 'Ashok Vihar', 'Aya Nagar', 
    'Daheli Sujanpur', 'Khirki Extension Panchsheel Vihar', 'C R Park', 'Chhattarpur Enclave Phase1', 'Bawana', 
    'West Punjabi Bagh', 'Burari', 'Shakti Nagar', 'Sarita Vihar', 'Sector 3 Rohini', 'Mandawali', 
    'Vinod Nagar East', 'Sector 22 Rohini', 'Sheikh Sarai Village', 'Shakurpur Colony', 'Nangloi', 
    'Nehru Place', 'Mayur Vihar', 'Lajpat Nagar Vinoba Puri', 'Block E Lajpat Nagar I', 'Nawada', 
    'Nangli Sakrawati', 'Sector 34 Rohini', 'Nirman Vihar', 'Chattarpur Enclave', 'Vasant Kunj Enclave', 
    'Mayur Vihar 1 Extension', 'Santnagar', 'Kasturba Gandhi Marg', 'Vipin Garden', 'West Patel Nagar Road', 
    'DDA Flat'
]

status_options = ["Unfurnished", "Semi-Furnished", "Furnished"]

# Inputs from user
house_type = st.selectbox("Select Housetype", house_type_list)
house_size = st.number_input("Enter House Size (in sqft)", min_value=150, max_value=14521)
selected_location = st.selectbox("Select a Location", locations)
city = st.selectbox("Enter the city", ['Delhi'])
numBathrooms = st.selectbox("Select the number of bathrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
SecurityDeposit = st.number_input("Enter the deposit amount (in INR)", min_value=0.0, max_value=11401010.0)
Status = st.selectbox("Select the property status", status_options)

# Load the machine learning model
model_path = r"C:\Users\nadip\Music\MACHINE LEARNING\ML Project\rfr.pkl"
model_2 = pickle.load(open(model_path, "rb"))

# Prepare input for prediction
input_data = [[house_type, house_size, selected_location, city, numBathrooms, SecurityDeposit, Status]]

# Submit button
if st.button("Submit"):
    # Make prediction
    result = model_2.predict(input_data)
    st.success(f"The predicted rental price of the house is ₹{result[0]:,.2f}")

# Optional: Show inputs for debugging
with st.expander("Show Inputs (Debug Info)"):
    st.write(f"House Type: {house_type}")
    st.write(f"Location: {selected_location}")
    st.write(f"City: {city}")
    st.write(f"Property Status: {Status}")
    st.write(f"Bathrooms: {numBathrooms}")
    st.write(f"Security Deposit: ₹{SecurityDeposit}")
    st.write(f"House Size: {house_size} sqft")
