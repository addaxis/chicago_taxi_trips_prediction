# import tensorflow_transform as tft
# import tensorflow as tf


# known_companies = ['Sun_Taxi', 'U_Taxicab', 'Blue_Ribbon_Taxi_Association', 'Flash_Cab', 'City_Service', 'Star_North_Taxi_Management_Llc', 'Taxi_Affiliation_Services', 'Top_Cab', 'Taxicab_Insurance_Agency__LLC', '5_Star_Taxi', 'Choice_Taxi_Association_Inc', 'Chicago_Independents', 'Globe_Taxi', 'Medallion_Leasin', 'Choice_Taxi_Association', 'Chicago_City_Taxi_Association', 'Taxicab_Insurance_Agency_Llc', 'Setare_Inc', 'Patriot_Taxi_Dba_Peace_Taxi_Associat', '5167___71969_5167_Taxi_Inc', 'KOAM_Taxi_Association', 'Metro_Jet_Taxi_A_', '312_Medallion_Management_Corp', 'Petani_Cab_Corp', 'Leonard_Cab_Co', 'Top_Cab_Affiliation', '4053___40193_Adwar_H__Nikola', 'Chicago_Taxicab', 'Taxi_Affiliation_Services_Llc___Yell', '2733___74600_Benny_Jona', '6574___Babylon_Express_Inc_', '3591___63480_Chuks_Cab', '3556___36214_RC_Andrews_Cab', '4623___27290_Jay_Kim', '4787___56058_Reny_Cab_Co', '5062___34841_Sam_Mestas']
# known_p_areas = [2.0, 56.0, 6.0, 14.0, 76.0, 1.0, 35.0, 10.0, 74.0, 70.0, 9.0, 32.0, 8.0, 5.0, 64.0, 25.0, 16.0, 68.0, 38.0, 77.0, 28.0, 52.0, 4.0, 39.0, 75.0, 3.0, 50.0, 24.0, 15.0, 67.0, 53.0, 12.0, 36.0, 29.0, 41.0, 55.0, 34.0, 33.0, 44.0, 47.0, 30.0, 7.0, 61.0, 72.0, 71.0, 11.0, 49.0, 37.0, 51.0, 40.0, 66.0, 48.0, 31.0, 22.0, 43.0, 69.0, 42.0, 45.0, 60.0, 46.0, 63.0, 54.0, 73.0, 65.0, 19.0, 23.0, 62.0, 13.0, 17.0, 21.0, 58.0, 57.0, 18.0, 27.0, 26.0, 59.0, 20.0]
# known_d_areas = [9.0, 74.0, 55.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 75.0, 76.0, 77.0]

import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    # Inputs would be a dictionary mapping feature names to Tensors or SparseTensors
    outputs = {}

    # Standardizing numeric columns
    outputs['standardized_trip_miles'] = tft.scale_to_z_score(inputs['trip_miles'])
    outputs['standardized_trip_seconds'] = tft.scale_to_z_score(inputs['trip_seconds'])

    # Extracting month and hour from timestamp
    outputs['trip_start_month'] = tf.strings.to_number(tf.strings.substr(inputs['trip_start_timestamp'], 5, 2), tf.int64)
    outputs['trip_start_hour'] = tf.strings.to_number(tf.strings.substr(inputs['trip_start_timestamp'], 11, 2), tf.int64)
    
    unique_companies = ['Flash_Cab',
 'Blue_Ribbon_Taxi_Association',
 'Taxicab_Insurance_Agency_Llc',
 'Taxi_Affiliation_Services',
 'Globe_Taxi',
 'Chicago_Independents',
 'City_Service',
 '5_Star_Taxi',
 'Patriot_Taxi_Dba_Peace_Taxi_Associat',
 'Sun_Taxi',
 'Medallion_Leasin',
 'Choice_Taxi_Association',
 'Star_North_Taxi_Management_Llc',
 'Taxicab_Insurance_Agency__LLC',
 'KOAM_Taxi_Association',
 'Choice_Taxi_Association_Inc',
 'Chicago_City_Taxi_Association',
 'U_Taxicab',
 'Top_Cab',
 'Taxi_Affiliation_Services_Llc___Yell',
 'Chicago_Taxicab',
 '3556___36214_RC_Andrews_Cab',
 'Top_Cab_Affiliation',
 '312_Medallion_Management_Corp',
 '5167___71969_5167_Taxi_Inc',
 'Metro_Jet_Taxi_A_',
 '3591___63480_Chuks_Cab',
 'Leonard_Cab_Co',
 '2733___74600_Benny_Jona',
 'Setare_Inc',
 '4053___40193_Adwar_H__Nikola',
 '6574___Babylon_Express_Inc_',
 'Petani_Cab_Corp',
 '4623___27290_Jay_Kim',
 '4787___56058_Reny_Cab_Co',
 '5062___34841_Sam_Mestas']
    
    unique_pickup_areas = [10,
 4,
 59,
 39,
 13,
 29,
 52,
 38,
 32,
 14,
 31,
 8,
 5,
 49,
 28,
 6,
 43,
 61,
 36,
 69,
 34,
 15,
 22,
 75,
 16,
 44,
 68,
 11,
 55,
 42,
 3,
 37,
 54,
 53,
 18,
 48,
 40,
 41,
 46,
 12,
 17,
 71,
 7,
 21,
 66,
 67,
 25,
 73,
 19,
 58,
 45,
 57,
 9,
 60,
 47,
 24,
 27,
 51,
 30,
 70,
 77,
 63,
 74,
 65,
 50,
 23,
 72,
 35,
 26,
 20,
 62,
 2,
 64,
 1,
 33,
 76,
 56]
    unique_dropoff_areas = [10,
 8,
 59,
 32,
 21,
 43,
 75,
 28,
 16,
 45,
 4,
 76,
 33,
 24,
 65,
 34,
 15,
 39,
 6,
 77,
 31,
 69,
 56,
 7,
 44,
 51,
 50,
 3,
 49,
 72,
 2,
 38,
 54,
 48,
 40,
 71,
 46,
 11,
 1,
 36,
 14,
 70,
 52,
 22,
 60,
 61,
 13,
 73,
 17,
 41,
 58,
 57,
 23,
 53,
 30,
 29,
 35,
 67,
 12,
 18,
 37,
 27,
 68,
 19,
 62,
 42,
 5,
 25,
 63,
 74,
 66,
 20,
 47,
 55,
 26,
 64,
 9]
    # One-hot encoding for categorical features
    # Note: You'll need to have the unique values for 'company', 'pickup_community_area', and 'dropoff_community_area' available
    for company in unique_companies:
        outputs[f'company_{company}'] = tf.cast(tf.equal(inputs['company'], company), tf.int64)

    for area in unique_pickup_areas:
        outputs[f'pickup_community_area_{area}'] = tf.cast(tf.equal(inputs['pickup_community_area'], area), tf.int64)

    for area in unique_dropoff_areas:
        outputs[f'dropoff_community_area_{area}'] = tf.cast(tf.equal(inputs['dropoff_community_area'], area), tf.int64)

    # Copying other required fields as is
    outputs['fare'] = inputs['fare']

    return outputs

