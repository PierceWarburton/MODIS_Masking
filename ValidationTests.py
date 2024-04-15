# Libraries
import pandas as pd
import numpy as np
import glob
from pyhdf.SD import SD, SDC
import cartopy as cartopy
cartopy.__path__.append('/usr/lib/python3.9/site-packages/cartopy/')
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cv2 as cv #pip3 install opencv-python
from pyresample.geometry import GridDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
from PIL import Image
import skimage
from scipy.spatial.distance import cdist
import netCDF4
import matplotlib.path as mplPath
import json

# General Function for sorting lists
def listsort(list):
    natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]
    sortedlist = sorted(list, key=natsort)
    return(sortedlist)

# Function for checking Proposed Dataset against Song et al. 2022
def ValidationCheckSong(i, Song_Tracks, labelfiles, maskedfiles): 
    file = labelfiles[i]

    timestamp = labelfiles[i].split('.')[1][1:] + '_' + labelfiles[i].split('.')[2]
    if timestamp not in list(Song_Tracks.keys()):
        sad = 'yes'
        return(pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks']))
    else:
        valid_track = np.copy(Song_Tracks[timestamp])
        reader = open(file)
        hdf = SD(file, SDC.READ)
        # Read dataset.
        data2D = hdf.select('Atm_Corr_Refl')
        np.shape(data2D)
        data = data2D[:,:].astype(np.double)
        #print(np.shape(data))
        # Read geolocation dataset.
        lat = hdf.select('Latitude')
        latitude = lat[:,:]
        lon = hdf.select('Longitude')
        longitude = lon[:,:]
        if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
            above_indices = longitude > 0.
            below_indices = longitude < 0.
            longitude[above_indices] = (longitude[above_indices]) - 180
            longitude[below_indices] = (longitude[below_indices]) + 180
        # Retrieve attributes.
        attrs = data2D.attributes(full=1)
        aoa=attrs["add_offset"]
        add_offset = aoa[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
        ua=attrs["units"]
        units = ua[0]
        data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        latitude_m = latitude
        #print(np.shape(latitude_m))
        longitude_m = longitude
        #print(np.shape(longitude_m))

        # Find middle location for plotting
        lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
        lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]

        # else:
            # data_m = np.vstack([data_m, datam])
            # latitude_m = np.vstack([latitude_m, latitude])
            # longitude_m = np.vstack([longitude_m, longitude])


        min_lon = np.min(longitude_m)
        max_lon = np.max(longitude_m)
        min_lat = np.min(latitude_m)
        max_lat = np.max(latitude_m)

        x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
        y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
        #data_ms = datam

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        #Have to flip shape because cv2 takes in coordinates y,x instead of x,y

        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon

        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 


        #### Now Mask Part
        scaled_label = np.asarray(Image.open(maskedfiles[i]))[:,:,1]
        data = np.copy(scaled_label)
        data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))
        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        #data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
        data_ms = datam

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        #Have to flip shape because cv2 takes in coordinates y,x instead of x,y
        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon
        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 
        result_imp[result_imp < 0.01] = 0
        result_imp[result_imp > 0.01] = 1

        # Split tracks up into seperate areas with magic
        labeled = skimage.morphology.label(result_imp)
        tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }

        # Calculate the mean of each array and compare it to the list of Yuan means
        # Also calculate the width of each track
        track_center = np.empty([len(tracks), 2])
        widths = []
        for num in tracks:
            track_center[num-1] = [np.mean(y[tracks[num][0]]),np.mean(x[tracks[num][1]])]

    for track in range(len(valid_track)):
        count = 0
        # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
        while (valid_track[track][1] > np.max(longitude_s) or valid_track[track][1] < np.min(longitude_s))\
            and count < 5:
            #print('Long: {}'.format(valid_track[track][1]))
            if Song_Tracks[timestamp][track][1] > 0:
                valid_track[track][1] -= 90
            else:
                valid_track[track][1] += 90 
            if valid_track[track][1] > 360:
                valid_track[track][1] -= 360
            if valid_track[track][1] < -360:
                valid_track[track][1] += 360
            # Set string to show a correction was made
            count += 1

    #for track in valid_track:
    output = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])
    
    if len(tracks) < 1:
        output['Timestamp'] = [timestamp] * len(tracks)
        output['NumPierceTracks'] = [0] * len(tracks)
        output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
        output['PierceTrack'] = ['NA'] * len(tracks)
        output['PointValid'] = [False] * len(tracks)
        output['PointValid'] = output['PointValid'].astype(bool)

    else:
        output['Timestamp'] = [timestamp] * len(tracks)
        output['NumPierceTracks'] = [len(tracks)] * len(tracks)
        output['NumTrueTracks'] = [len(valid_track)] * len(tracks)

        # See if center falls into my Tracks
        valid_center = []
        matched_tracks = 0
        centercount = 1
        centerlist = []
        for num in tracks:
            centerlist.append(centercount)
            valid = False
            piercetrack = np.stack([y[tracks[num][0]], x[tracks[num][1]]], axis =1)
            for j in range(len(valid_track)):
                dis = cdist(piercetrack, valid_track[j:j+1])
                if np.min(dis) < 0.5:
                    valid = True
                    matched_tracks += 1
            valid_center.append(valid)

            centercount += 1

        output['PierceTrack'] = centerlist
        output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
        output['PointValid'] = valid_center
        output['PointValid'] = output['PointValid'].astype(bool)

    return(output)


    validationBoth = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])
    for timestamp in Song_Tracks:
        tracks = {}
        if timestamp in Toll_Centers.keys():
            for i in range(1,len(Toll_Tracks[timestamp])+1):
                tracks[i] = (np.array(Toll_Tracks[timestamp][i-1])[:,0],np.array(Toll_Tracks[timestamp][i-1])[:,1])
            output = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])

            valid_track = Song_Tracks[timestamp]
            # Get lat/lon limits
            for file in labelfiles:
                if timestamp.replace('_', '.') in file:
                    reader = open(file)
                    hdf = SD(file, SDC.READ)
                    # Read dataset.
                    data2D = hdf.select('Atm_Corr_Refl')
                    data = data2D[:,:].astype(np.double)
                    # Read geolocation dataset.
                    lat = hdf.select('Latitude')
                    latitude = lat[:,:]
                    lon = hdf.select('Longitude')
                    longitude = lon[:,:]
                    min_lon = np.min(longitude)
                    max_lon = np.max(longitude)
                    # Retrieve attributes.
                    attrs = data2D.attributes(full=1)
                    aoa=attrs["add_offset"]
                    add_offset = aoa[0]
                    fva=attrs["_FillValue"]
                    _FillValue = fva[0]
                    sfa=attrs["scale_factor"]
                    scale_factor = sfa[0]        
                    ua=attrs["units"]
                    units = ua[0]
                    data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
                    datam = (data - add_offset) * scale_factor 
                    data_m = np.ma.masked_array(datam, np.isnan(datam))
                    #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
                    mask = data_m.mask
                    datam = data_m.data 
                    data_ms = datam#[:,:,4] #pick band (5th band shows ship tracks)
                    #data_ms = datam

                    longitude_s = cv.resize(longitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                    latitude_s = cv.resize(latitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)


            # Correct the Song stuff
            for track in range(len(valid_track)):
                count = 0
                # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
                while (valid_track[track][1] > max_lon or valid_track[track][1] < min_lon)\
                    and count < 5:
                    #print('Long: {}'.format(valid_track[track][1]))
                    if Song_Tracks[timestamp][track][1] > 0:
                        valid_track[track][1] -= 90
                    else:
                        valid_track[track][1] += 90 
                    if valid_track[track][1] > 360:
                        valid_track[track][1] -= 360
                    if valid_track[track][1] < -360:
                        valid_track[track][1] += 360
                    # Set string to show a correction was made
                    count += 1


                #for track in valid_tracks
                
                if len(tracks) < 1:
                    output['Timestamp'] = [timestamp] * len(tracks)
                    output['NumTestTracks'] = [0] * len(tracks)
                    output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
                    output['TestTrack'] = ['NA'] * len(tracks)
                    output['PointValid'] = [False] * len(tracks)
                    output['PointValid'] = output['PointValid'].astype(bool)

                else:
                    output['Timestamp'] = [timestamp] * len(tracks)
                    output['NumTestTracks'] = [len(tracks)] * len(tracks)
                    output['NumTrueTracks'] = [len(valid_track)] * len(tracks)

                    # See if center falls into my Tracks
                    valid_center = []
                    matched_tracks = 0
                    centercount = 1
                    centerlist = []
                    for num in tracks:
                        centerlist.append(centercount)
                        valid = False
                        testtrack = np.stack([tracks[num][0], tracks[num][1]], axis =1)
                        for j in range(len(valid_track)):
                            dis = cdist(testtrack, valid_track[j:j+1])
                            if np.min(dis) < 0.5:
                                valid = True
                                matched_tracks += 1
                        valid_center.append(valid)

                        centercount += 1

                    output['TestTrack'] = centerlist
                    output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
                    output['PointValid'] = valid_center
                    output['PointValid'] = output['PointValid'].astype(bool)

        validationBoth = pd.concat([validationBoth, output])
        validationBoth['PointValid'] = validationBoth['PointValid'].astype(bool)
        
        return(validationBoth)

# Function for checking the Proposed Dataset against Toll et al. 2019
def ValidationCheckToll(i, Toll_Tracks, labelfiles, maskedfiles):
        # for file in labelfiles: #Pierce to add for loop 
    file = labelfiles[i]

    timestamp = labelfiles[i].split('.')[1][1:] + '_' + labelfiles[i].split('.')[2]
    if timestamp not in list(Toll_Tracks.keys()):
        sad = 'yes'
        return(pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks']))
    else:
        valid_track = np.copy(Toll_Tracks[timestamp])
        reader = open(file)
        hdf = SD(file, SDC.READ)
        # Read dataset.
        data2D = hdf.select('Atm_Corr_Refl')
        np.shape(data2D)
        data = data2D[:,:].astype(np.double)
        #print(np.shape(data))
        # Read geolocation dataset.
        lat = hdf.select('Latitude')
        latitude = lat[:,:]
        lon = hdf.select('Longitude')
        longitude = lon[:,:]
        if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
            above_indices = longitude > 0.
            below_indices = longitude < 0.
            longitude[above_indices] = (longitude[above_indices]) - 180
            longitude[below_indices] = (longitude[below_indices]) + 180
        # Retrieve attributes.
        attrs = data2D.attributes(full=1)
        aoa=attrs["add_offset"]
        add_offset = aoa[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
        ua=attrs["units"]
        units = ua[0]
        data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        latitude_m = latitude
        #print(np.shape(latitude_m))
        longitude_m = longitude
        #print(np.shape(longitude_m))

        # Find middle location for plotting
        lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
        lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]

        # else:
            # data_m = np.vstack([data_m, datam])
            # latitude_m = np.vstack([latitude_m, latitude])
            # longitude_m = np.vstack([longitude_m, longitude])


        min_lon = np.min(longitude_m)
        max_lon = np.max(longitude_m)
        min_lat = np.min(latitude_m)
        max_lat = np.max(latitude_m)

        x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
        y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
        #data_ms = datam

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        #Have to flip shape because cv2 takes in coordinates y,x instead of x,y

        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon

        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 


        #### Now Mask Part
        scaled_label = np.asarray(Image.open(maskedfiles[i]))[:,:,1]
        data = np.copy(scaled_label)
        data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))
        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        #data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
        data_ms = datam

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        #Have to flip shape because cv2 takes in coordinates y,x instead of x,y
        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon
        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 
        result_imp[result_imp < 0.01] = 0
        result_imp[result_imp > 0.01] = 1

        # Split tracks up into seperate areas with magic
        labeled = skimage.morphology.label(result_imp)
        tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }

        # Calculate the mean of each array and compare it to the list of Yuan means
        # Also calculate the width of each track
        track_center = np.empty([len(tracks), 2])
        widths = []
        for num in tracks:
            track_center[num-1] = [np.mean(y[tracks[num][0]]),np.mean(x[tracks[num][1]])]

    for track in range(len(valid_track)):
        count = 0
        # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
        while (valid_track[track][1] > np.max(longitude_s) or valid_track[track][1] < np.min(longitude_s))\
            and count < 5:
            #print('Long: {}'.format(valid_track[track][1]))
            if Toll_Tracks[timestamp][track][1] > 0:
                valid_track[track][1] -= 90
            else:
                valid_track[track][1] += 90 
            if valid_track[track][1] > 360:
                valid_track[track][1] -= 360
            if valid_track[track][1] < -360:
                valid_track[track][1] += 360
            # Set string to show a correction was made
            count += 1

    #for track in valid_track:
    output = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])
    
    if len(tracks) < 1:
        output['Timestamp'] = [timestamp] * len(tracks)
        output['NumPierceTracks'] = [0] * len(tracks)
        output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
        output['PierceTrack'] = ['NA'] * len(tracks)
        output['PointValid'] = [False] * len(tracks)
        output['PointValid'] = output['PointValid'].astype(bool)

    else:
        output['Timestamp'] = [timestamp] * len(tracks)
        output['NumPierceTracks'] = [len(tracks)] * len(tracks)
        output['NumTrueTracks'] = [len(valid_track)] * len(tracks)

        # See if center falls into my Tracks
        valid_center = []
        matched_tracks = 0
        centercount = 1
        centerlist = []
        for num in tracks:
            centerlist.append(centercount)
            valid = False
            piercetrack = np.stack([y[tracks[num][0]], x[tracks[num][1]]], axis =1)
            for j in range(len(valid_track)):
                dis = cdist(piercetrack, valid_track[j:j+1])
                if np.min(dis) < 0.5:
                    valid = True
                    matched_tracks += 1
            valid_center.append(valid)

            centercount += 1

        output['PierceTrack'] = centerlist
        output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
        output['PointValid'] = valid_center
        output['PointValid'] = output['PointValid'].astype(bool)

    return(output)

# Function for checking Proposed Dataset against Watson-Parris et al. 2022
def ValidationCheckWatson_Parris(Watson_Parris_Tracks, maskedfiles):
    validationWatson_Parris = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumDunTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])

    i = 0
    for ts in Watson_Parris_Tracks:
        i += 1
        for myfile in maskedfiles:
            if ts in myfile:
                mine = np.array(Image.open(myfile))[:,:,1]
                mine[mine <= 1] = 0
                mine[mine > 1] = 1
        # Seperate my tracks
        labeled = skimage.morphology.label(mine)
        tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }
        # Find the center of my tracks
        track_center = np.empty([len(tracks), 2])
        for num in tracks:
            track_center[num-1] = [np.mean(tracks[num][0]),np.mean(tracks[num][1])]
        # Set up dataframe
        output = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumDunTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])
        output['Timestamp'] = [ts] * len(tracks)
        output['NumPierceTracks'] = [len(tracks)] * len(tracks)
        output['NumDunTracks'] = [len(Watson_Parris_Tracks[ts])] * len(tracks)

        # See if center falls into my Tracks
        valid_center = []
        matched_tracks = 0
        centercount = 1
        centerlist = []
        for num in tracks:
            centerlist.append(centercount)
            #poly_path = mplPath.Path(np.stack([tracks[num][0], tracks[num][1]], axis = 1))
            valid = False
            for labeltrack in Watson_Parris_Tracks[ts]:
                center = np.array(labeltrack)[len(labeltrack)//8]
                if (center.astype(int)[0] in tracks[num][1]) and (center.astype(int)[1] in tracks[num][0]):
                    valid = True
                    matched_tracks += 1
            valid_center.append(valid)
            centercount += 1

        output['PierceTrack'] = centerlist
        output['UnMatchedTrueTracks'] = [len(Watson_Parris_Tracks[ts])- sum(valid_center)] * len(tracks)
        output['PointValid'] = valid_center
        output['PointValid'] = output['PointValid'].astype(bool)
        validationWatson_Parris = pd.concat([validationWatson_Parris, output])
        validationWatson_Parris['PointValid'] = validationWatson_Parris['PointValid'].astype(bool)

    return(validationWatson_Parris)

# Function for checking Watson-Parris et al. 2022 against Toll et al. 2019
def ValidationCheckWatsonParris_Toll(timestamp, Toll_Centers, Watson_Parris_Tracks, labelfiles):
    if timestamp not in Watson_Parris_Tracks.keys():
        return(pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks']))
    else:
        output = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])
        valid_track = Toll_Centers[timestamp]

        # Get full mask track for the DunWat stuff
        image = np.zeros([2030, 1354])
        for track in Watson_Parris_Tracks[timestamp]:
            test = np.stack([np.array(track)[:,1], np.array(track)[:,0]], axis=1).astype(int)
            poly_path = mplPath.Path(test)
            for i in range(len(image)):
                for j in range(len(image[0])):
                    if poly_path.contains_point([i,j]):
                        image[i,j] = 1
                        
        # Get lat/lon limits
        for file in labelfiles:
            if timestamp.replace('_', '.') in file:
                reader = open(file)
                hdf = SD(file, SDC.READ)
                # Read dataset.
                data2D = hdf.select('Atm_Corr_Refl')
                np.shape(data2D)
                data = data2D[:,:].astype(np.double)
                #print(np.shape(data))
                # Read geolocation dataset.
                lat = hdf.select('Latitude')
                latitude = lat[:,:]
                lon = hdf.select('Longitude')
                longitude = lon[:,:]
                if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
                    above_indices = longitude > 0.
                    below_indices = longitude < 0.
                    longitude[above_indices] = (longitude[above_indices]) - 180
                    longitude[below_indices] = (longitude[below_indices]) + 180
                # Retrieve attributes.
                attrs = data2D.attributes(full=1)
                aoa=attrs["add_offset"]
                add_offset = aoa[0]
                fva=attrs["_FillValue"]
                _FillValue = fva[0]
                sfa=attrs["scale_factor"]
                scale_factor = sfa[0]        
                ua=attrs["units"]
                units = ua[0]
                data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
                datam = (data - add_offset) * scale_factor 
                data_m = np.ma.masked_array(datam, np.isnan(datam))

                latitude_m = latitude
                #print(np.shape(latitude_m))
                longitude_m = longitude
                #print(np.shape(longitude_m))

                # Find middle location for plotting
                lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
                lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]

                # else:
                    # data_m = np.vstack([data_m, datam])
                    # latitude_m = np.vstack([latitude_m, latitude])
                    # longitude_m = np.vstack([longitude_m, longitude])


                min_lon = np.min(longitude_m)
                max_lon = np.max(longitude_m)
                min_lat = np.min(latitude_m)
                max_lat = np.max(latitude_m)

                x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
                y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


                #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
                mask = data_m.mask
                datam = data_m.data 
                data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
                #data_ms = datam

                longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                #Have to flip shape because cv2 takes in coordinates y,x instead of x,y

                #Define swath and project to raw lat/lon 
                swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
                cellSize = 0.01
                x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
                nx = int(np.floor((max_lon - min_lon) / cellSize))
                ny = int(np.floor((max_lat - min_lat) / cellSize))
                x = np.linspace(x0, x0 + xinc*nx, nx)
                y = np.linspace(y0, y0 + yinc*ny, ny)
                lon_g, lat_g = np.meshgrid(x, y)
                grid_def = GridDefinition(lons=lon_g, lats=lat_g)
                np.shape(grid_def)
                ri = 10000 #play around with this to fill image, this value seems to work well 
                result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon

                #IMPUTE NANs with minimum value across non-nan pixels 
                result_imp = np.where(np.isnan(result), np.nanmin(result), result) 


                #### Now Mask Part
                #scaled_label = np.asarray(Image.open(fadedfiles[i]))[:,:,1]
                data = image
                data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
                datam = (data - add_offset) * scale_factor 
                data_m = np.ma.masked_array(datam, np.isnan(datam))
                #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
                mask = data_m.mask
                datam = data_m.data 
                #data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
                data_ms = datam

                longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                #Have to flip shape because cv2 takes in coordinates y,x instead of x,y
                #Define swath and project to raw lat/lon 
                swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
                cellSize = 0.01
                x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
                nx = int(np.floor((max_lon - min_lon) / cellSize))
                ny = int(np.floor((max_lat - min_lat) / cellSize))
                x = np.linspace(x0, x0 + xinc*nx, nx)
                y = np.linspace(y0, y0 + yinc*ny, ny)
                lon_g, lat_g = np.meshgrid(x, y)
                grid_def = GridDefinition(lons=lon_g, lats=lat_g)
                np.shape(grid_def)
                ri = 10000 #play around with this to fill image, this value seems to work well 
                result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon
                #IMPUTE NANs with minimum value across non-nan pixels 
                result_imp = np.where(np.isnan(result), np.nanmin(result), result) 
                buffer = 0.00005

                result_imp[result_imp < buffer] = 0
                result_imp[result_imp > buffer] = 1


        # Correct the Song stuff
        for track in range(len(valid_track)):
            count = 0
            # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
            while (valid_track[track][1] > max_lon or valid_track[track][1] < min_lon)\
                and count < 5:
                #print('Long: {}'.format(valid_track[track][1]))
                if validation_tracks[timestamp][track][1] > 0:
                    valid_track[track][1] -= 90
                else:
                    valid_track[track][1] += 90 
                if valid_track[track][1] > 360:
                    valid_track[track][1] -= 360
                if valid_track[track][1] < -360:
                    valid_track[track][1] += 360
                # Set string to show a correction was made
                count += 1

            labeled = skimage.morphology.label(result_imp)
            tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }


            #for track in valid_tracks
            
            if len(DunWat[timestamp]) < 1:
                output['Timestamp'] = [timestamp] * len(tracks)
                output['NumTestTracks'] = [0] * len(tracks)
                output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
                output['TestTrack'] = ['NA'] * len(tracks)
                output['PointValid'] = [False] * len(tracks)
                output['PointValid'] = output['PointValid'].astype(bool)

            else:
                output['Timestamp'] = [timestamp] * len(tracks)
                output['NumTestTracks'] = [len(DunWat[timestamp])] * len(tracks)
                output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
                
                labeled = skimage.morphology.label(result_imp)
                tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }

                # See if center falls into my Tracks
                valid_center = []
                matched_tracks = 0
                centercount = 1
                centerlist = []
                for num in tracks:
                    testtrack = np.stack([np.array(tracks[num])[1,:], np.array(tracks[num])[0,:]], axis = 1)
                    centerlist.append(centercount)
                    valid = False
                    for j in range(len(valid_track)):
                        a = np.unravel_index((np.abs(y - valid_track[j][0])).argmin(), y.shape)
                        b = np.unravel_index((np.abs(x - valid_track[j][1])).argmin(), x.shape)
                        temp = np.empty([2,2])
                        temp[0]= (b[0],a[0])
                        dis = cdist(testtrack, temp[0:1])
                        if np.min(dis) < 20:
                            valid = True
                            matched_tracks += 1
                    valid_center.append(valid)

                    centercount += 1

                output['TestTrack'] = centerlist
                output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
                output['PointValid'] = valid_center
                output['PointValid'] = output['PointValid'].astype(bool)
    return(output)


        # for file in labelfiles: #Pierce to add for loop 
    file = labelfiles[i]

    timestamp = labelfiles[i].split('.')[1][1:] + '_' + labelfiles[i].split('.')[2]
    if timestamp not in list(Toll_Tracks.keys()):
        sad = 'yes'
        return(pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks']))
    else:
        valid_track = np.copy(Toll_Tracks[timestamp])
        reader = open(file)
        hdf = SD(file, SDC.READ)
        # Read dataset.
        data2D = hdf.select('Atm_Corr_Refl')
        np.shape(data2D)
        data = data2D[:,:].astype(np.double)
        #print(np.shape(data))
        # Read geolocation dataset.
        lat = hdf.select('Latitude')
        latitude = lat[:,:]
        lon = hdf.select('Longitude')
        longitude = lon[:,:]
        if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
            above_indices = longitude > 0.
            below_indices = longitude < 0.
            longitude[above_indices] = (longitude[above_indices]) - 180
            longitude[below_indices] = (longitude[below_indices]) + 180
        # Retrieve attributes.
        attrs = data2D.attributes(full=1)
        aoa=attrs["add_offset"]
        add_offset = aoa[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
        ua=attrs["units"]
        units = ua[0]
        data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        latitude_m = latitude
        #print(np.shape(latitude_m))
        longitude_m = longitude
        #print(np.shape(longitude_m))

        # Find middle location for plotting
        lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
        lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]

        # else:
            # data_m = np.vstack([data_m, datam])
            # latitude_m = np.vstack([latitude_m, latitude])
            # longitude_m = np.vstack([longitude_m, longitude])


        min_lon = np.min(longitude_m)
        max_lon = np.max(longitude_m)
        min_lat = np.min(latitude_m)
        max_lat = np.max(latitude_m)

        x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
        y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
        #data_ms = datam

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        #Have to flip shape because cv2 takes in coordinates y,x instead of x,y

        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon

        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 


        #### Now Mask Part
        scaled_label = np.asarray(Image.open(maskedfiles[i]))[:,:,1]
        data = np.copy(scaled_label)
        data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))
        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        #data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
        data_ms = datam

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        #Have to flip shape because cv2 takes in coordinates y,x instead of x,y
        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon
        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 
        result_imp[result_imp < 0.01] = 0
        result_imp[result_imp > 0.01] = 1

        # Split tracks up into seperate areas with magic
        labeled = skimage.morphology.label(result_imp)
        tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }

        # Calculate the mean of each array and compare it to the list of Yuan means
        # Also calculate the width of each track
        track_center = np.empty([len(tracks), 2])
        widths = []
        for num in tracks:
            track_center[num-1] = [np.mean(y[tracks[num][0]]),np.mean(x[tracks[num][1]])]

    for track in range(len(valid_track)):
        count = 0
        # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
        while (valid_track[track][1] > np.max(longitude_s) or valid_track[track][1] < np.min(longitude_s))\
            and count < 5:
            #print('Long: {}'.format(valid_track[track][1]))
            if Toll_Tracks[timestamp][track][1] > 0:
                valid_track[track][1] -= 90
            else:
                valid_track[track][1] += 90 
            if valid_track[track][1] > 360:
                valid_track[track][1] -= 360
            if valid_track[track][1] < -360:
                valid_track[track][1] += 360
            # Set string to show a correction was made
            count += 1

    #for track in valid_track:
    output = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])
    
    if len(tracks) < 1:
        output['Timestamp'] = [timestamp] * len(tracks)
        output['NumPierceTracks'] = [0] * len(tracks)
        output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
        output['PierceTrack'] = ['NA'] * len(tracks)
        output['PointValid'] = [False] * len(tracks)
        output['PointValid'] = output['PointValid'].astype(bool)

    else:
        output['Timestamp'] = [timestamp] * len(tracks)
        output['NumPierceTracks'] = [len(tracks)] * len(tracks)
        output['NumTrueTracks'] = [len(valid_track)] * len(tracks)

        # See if center falls into my Tracks
        valid_center = []
        matched_tracks = 0
        centercount = 1
        centerlist = []
        for num in tracks:
            centerlist.append(centercount)
            valid = False
            piercetrack = np.stack([y[tracks[num][0]], x[tracks[num][1]]], axis =1)
            for j in range(len(valid_track)):
                dis = cdist(piercetrack, valid_track[j:j+1])
                if np.min(dis) < 0.5:
                    valid = True
                    matched_tracks += 1
            valid_center.append(valid)

            centercount += 1

        output['PierceTrack'] = centerlist
        output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
        output['PointValid'] = valid_center
        output['PointValid'] = output['PointValid'].astype(bool)

    return(output)

# Function for checking Song et al. 2022 against Toll et al. 2019
def ValidationCheckSongAndToll(Song_Tracks, Toll_Tracks, Toll_Centers, labelfiles):
    validationBoth = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])
    for timestamp in Song_Tracks:
        tracks = {}
        if timestamp in Toll_Centers.keys():
            for i in range(1,len(Toll_Tracks[timestamp])+1):
                tracks[i] = (np.array(Toll_Tracks[timestamp][i-1])[:,0],np.array(Toll_Tracks[timestamp][i-1])[:,1])
            output = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])

            valid_track = Song_Tracks[timestamp]
            # Get lat/lon limits
            for file in labelfiles:
                if timestamp.replace('_', '.') in file:
                    reader = open(file)
                    hdf = SD(file, SDC.READ)
                    # Read dataset.
                    data2D = hdf.select('Atm_Corr_Refl')
                    data = data2D[:,:].astype(np.double)
                    # Read geolocation dataset.
                    lat = hdf.select('Latitude')
                    latitude = lat[:,:]
                    lon = hdf.select('Longitude')
                    longitude = lon[:,:]
                    min_lon = np.min(longitude)
                    max_lon = np.max(longitude)
                    # Retrieve attributes.
                    attrs = data2D.attributes(full=1)
                    aoa=attrs["add_offset"]
                    add_offset = aoa[0]
                    fva=attrs["_FillValue"]
                    _FillValue = fva[0]
                    sfa=attrs["scale_factor"]
                    scale_factor = sfa[0]        
                    ua=attrs["units"]
                    units = ua[0]
                    data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
                    datam = (data - add_offset) * scale_factor 
                    data_m = np.ma.masked_array(datam, np.isnan(datam))
                    #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
                    mask = data_m.mask
                    datam = data_m.data 
                    data_ms = datam#[:,:,4] #pick band (5th band shows ship tracks)
                    #data_ms = datam

                    longitude_s = cv.resize(longitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                    latitude_s = cv.resize(latitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)


            # Correct the Song stuff
            for track in range(len(valid_track)):
                count = 0
                # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
                while (valid_track[track][1] > max_lon or valid_track[track][1] < min_lon)\
                    and count < 5:
                    #print('Long: {}'.format(valid_track[track][1]))
                    if Song_Tracks[timestamp][track][1] > 0:
                        valid_track[track][1] -= 90
                    else:
                        valid_track[track][1] += 90 
                    if valid_track[track][1] > 360:
                        valid_track[track][1] -= 360
                    if valid_track[track][1] < -360:
                        valid_track[track][1] += 360
                    # Set string to show a correction was made
                    count += 1


                #for track in valid_tracks
                
                if len(tracks) < 1:
                    output['Timestamp'] = [timestamp] * len(tracks)
                    output['NumTestTracks'] = [0] * len(tracks)
                    output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
                    output['TestTrack'] = ['NA'] * len(tracks)
                    output['PointValid'] = [False] * len(tracks)
                    output['PointValid'] = output['PointValid'].astype(bool)

                else:
                    output['Timestamp'] = [timestamp] * len(tracks)
                    output['NumTestTracks'] = [len(tracks)] * len(tracks)
                    output['NumTrueTracks'] = [len(valid_track)] * len(tracks)

                    # See if center falls into my Tracks
                    valid_center = []
                    matched_tracks = 0
                    centercount = 1
                    centerlist = []
                    for num in tracks:
                        centerlist.append(centercount)
                        valid = False
                        testtrack = np.stack([tracks[num][0], tracks[num][1]], axis =1)
                        for j in range(len(valid_track)):
                            dis = cdist(testtrack, valid_track[j:j+1])
                            if np.min(dis) < 0.5:
                                valid = True
                                matched_tracks += 1
                        valid_center.append(valid)

                        centercount += 1

                    output['TestTrack'] = centerlist
                    output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
                    output['PointValid'] = valid_center
                    output['PointValid'] = output['PointValid'].astype(bool)

        validationBoth = pd.concat([validationBoth, output])
        validationBoth['PointValid'] = validationBoth['PointValid'].astype(bool)
        
        return(validationBoth)

# Function for checking Song et al. 2022 against Watson-Parris et al. 2022
def ValidationCheckSong_WatsonParris(timestamp, Song_Tracks, Watson_Parris_Tracks):
    if timestamp not in Watson_Parris_Tracks.keys():
        return(pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks']))
    else:
        output = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])
        valid_track = Song_Tracks[timestamp]

        # Get full mask track for the DunWat stuff
        image = np.zeros([2030, 1354])
        for track in Watson_Parris_Tracks[timestamp]:
            test = np.stack([np.array(track)[:,1], np.array(track)[:,0]], axis=1).astype(int)
            poly_path = mplPath.Path(test)
            for i in range(len(image)):
                for j in range(len(image[0])):
                    if poly_path.contains_point([i,j]):
                        image[i,j] = 1
                        
        # Get lat/lon limits
        for file in labelfiles:
            if timestamp.replace('_', '.') in file:
                reader = open(file)
                hdf = SD(file, SDC.READ)
                # Read dataset.
                data2D = hdf.select('Atm_Corr_Refl')
                np.shape(data2D)
                data = data2D[:,:].astype(np.double)
                #print(np.shape(data))
                # Read geolocation dataset.
                lat = hdf.select('Latitude')
                latitude = lat[:,:]
                lon = hdf.select('Longitude')
                longitude = lon[:,:]
                if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
                    above_indices = longitude > 0.
                    below_indices = longitude < 0.
                    longitude[above_indices] = (longitude[above_indices]) - 180
                    longitude[below_indices] = (longitude[below_indices]) + 180
                # Retrieve attributes.
                attrs = data2D.attributes(full=1)
                aoa=attrs["add_offset"]
                add_offset = aoa[0]
                fva=attrs["_FillValue"]
                _FillValue = fva[0]
                sfa=attrs["scale_factor"]
                scale_factor = sfa[0]        
                ua=attrs["units"]
                units = ua[0]
                data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
                datam = (data - add_offset) * scale_factor 
                data_m = np.ma.masked_array(datam, np.isnan(datam))

                latitude_m = latitude
                #print(np.shape(latitude_m))
                longitude_m = longitude
                #print(np.shape(longitude_m))

                # Find middle location for plotting
                lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
                lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]

                min_lon = np.min(longitude_m)
                max_lon = np.max(longitude_m)
                min_lat = np.min(latitude_m)
                max_lat = np.max(latitude_m)

                x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
                y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


                #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
                mask = data_m.mask
                datam = data_m.data 
                data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
                #data_ms = datam

                longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                #Have to flip shape because cv2 takes in coordinates y,x instead of x,y

                #Define swath and project to raw lat/lon 
                swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
                cellSize = 0.01
                x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
                nx = int(np.floor((max_lon - min_lon) / cellSize))
                ny = int(np.floor((max_lat - min_lat) / cellSize))
                x = np.linspace(x0, x0 + xinc*nx, nx)
                y = np.linspace(y0, y0 + yinc*ny, ny)
                lon_g, lat_g = np.meshgrid(x, y)
                grid_def = GridDefinition(lons=lon_g, lats=lat_g)
                np.shape(grid_def)
                ri = 10000 #play around with this to fill image, this value seems to work well 
                result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon

                #IMPUTE NANs with minimum value across non-nan pixels 
                result_imp = np.where(np.isnan(result), np.nanmin(result), result) 


                #### Now Mask Part
                #scaled_label = np.asarray(Image.open(fadedfiles[i]))[:,:,1]
                data = image
                data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
                datam = (data - add_offset) * scale_factor 
                data_m = np.ma.masked_array(datam, np.isnan(datam))
                #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
                mask = data_m.mask
                datam = data_m.data 
                #data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)
                data_ms = datam

                longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
                #Have to flip shape because cv2 takes in coordinates y,x instead of x,y
                #Define swath and project to raw lat/lon 
                swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
                cellSize = 0.01
                x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
                nx = int(np.floor((max_lon - min_lon) / cellSize))
                ny = int(np.floor((max_lat - min_lat) / cellSize))
                x = np.linspace(x0, x0 + xinc*nx, nx)
                y = np.linspace(y0, y0 + yinc*ny, ny)
                lon_g, lat_g = np.meshgrid(x, y)
                grid_def = GridDefinition(lons=lon_g, lats=lat_g)
                np.shape(grid_def)
                ri = 10000 #play around with this to fill image, this value seems to work well 
                result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon
                #IMPUTE NANs with minimum value across non-nan pixels 
                result_imp = np.where(np.isnan(result), np.nanmin(result), result) 
                buffer = 0.00005

                result_imp[result_imp < buffer] = 0
                result_imp[result_imp > buffer] = 1


        # Correct the Yuan stuff
        for track in range(len(valid_track)):
            count = 0
            # Check to see if the longitude values of Yuan tracks are within the right range and correct them if not
            while (valid_track[track][1] > max_lon or valid_track[track][1] < min_lon)\
                and count < 5:
                #print('Long: {}'.format(valid_track[track][1]))
                if Song_Tracks[timestamp][track][1] > 0:
                    valid_track[track][1] -= 90
                else:
                    valid_track[track][1] += 90 
                if valid_track[track][1] > 360:
                    valid_track[track][1] -= 360
                if valid_track[track][1] < -360:
                    valid_track[track][1] += 360
                # Set string to show a correction was made
                count += 1

            labeled = skimage.morphology.label(result_imp)
            tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }


            #for track in valid_tracks
            
            if len(Watson_Parris_Tracks[timestamp]) < 1:
                output['Timestamp'] = [timestamp] * len(tracks)
                output['NumTestTracks'] = [0] * len(tracks)
                output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
                output['TestTrack'] = ['NA'] * len(tracks)
                output['PointValid'] = [False] * len(tracks)
                output['PointValid'] = output['PointValid'].astype(bool)

            else:
                output['Timestamp'] = [timestamp] * len(tracks)
                output['NumTestTracks'] = [len(Watson_Parris_Tracks[timestamp])] * len(tracks)
                output['NumTrueTracks'] = [len(valid_track)] * len(tracks)
                
                labeled = skimage.morphology.label(result_imp)
                tracks = { i: (labeled == i).nonzero() for i in range(1,labeled.max()+1) }

                # See if center falls into my Tracks
                valid_center = []
                matched_tracks = 0
                centercount = 1
                centerlist = []
                for num in tracks:
                    testtrack = np.stack([np.array(tracks[num])[1,:], np.array(tracks[num])[0,:]], axis = 1)
                    centerlist.append(centercount)
                    valid = False
                    for j in range(len(valid_track)):
                        a = np.unravel_index((np.abs(y - valid_track[j][0])).argmin(), y.shape)
                        b = np.unravel_index((np.abs(x - valid_track[j][1])).argmin(), x.shape)
                        temp = np.empty([2,2])
                        temp[0]= (b[0],a[0])
                        dis = cdist(testtrack, temp[0:1])
                        if np.min(dis) < 20:
                            valid = True
                            matched_tracks += 1
                    valid_center.append(valid)

                    centercount += 1

                output['TestTrack'] = centerlist
                output['UnMatchedTrueTracks'] = [len(valid_track)- sum(valid_center)] * len(tracks)
                output['PointValid'] = valid_center
                output['PointValid'] = output['PointValid'].astype(bool)
    return(output)






### Load in Proposed Data
filepath = "path_to_Modis_Data/"
labelfiles = glob.glob(filepath + "*.hdf")
listsort(labelfiles)
filepath = 'paht_to_proposed_images/'
maskedfiles = glob.glob(filepath + "*_Masked.png")
listsort(maskedfiles)



### Load in Song et al. 2022 data
data = netCDF4.Dataset('path_to_song_data','r')

filepath = "path_to_proposed_images/"
files = glob.glob(filepath + "*Masked.png")
listsort(files)
file_timestamp = []
for file in files:
    file_timestamp.append(file.split('Images/')[1].split('_M')[0])

# Near the begining and end of proposed datasets timestamps
start = 110000
end = 180000

days = data.variables['date_doy'][start:end]
times = data.variables['time_utc'][start:end]

Song_Tracks = {}
for i in range(len(days)):
    curr = days[i] + '_' + times[i]
    # if curr == timestamp:
    #     saved = i
    if curr in file_timestamp:
        lat_mean = float(data.variables['lat_mean'][i+start])
        lon_mean = float(data.variables['lon_mean'][i+start])
        if curr not in list(Song_Tracks.keys()):
            Song_Tracks[curr] = []
        Song_Tracks[curr].append([lat_mean, lon_mean])



### Load in Watson-Parris et al. 2022
filepath = "path_to_Watson_Parris_files"
labelfiles = glob.glob(filepath + '*.json')
listsort(labelfiles)
imagefiles = glob.glob(filepath + '*.png')
listsort(imagefiles)

# Get points
match = []
for file in labelfiles:
    ts = file.split('myd')[-1].replace('.', '_').split('D_j')[0]
    if ts in file_timestamp:
        match.append(file)

# Get Images
imgmatch= []
for file in imagefiles:
    ts = file.split('d')[-1].replace('.', '_').split('D_p')[0]
    if ts in file_timestamp:
        imgmatch.append(file)

# Put Points in good format
Watson_Parris_Tracks = {}
for file in match:
    ts = file.split('myd')[-1].replace('.', '_').split('D_j')[0]
    Watson_Parris_Tracks[ts] = []
    with open(file) as f:
        data = json.load(f)
        for i in range(len(data['shapes'])):
            Watson_Parris_Tracks[ts].append(data['shapes'][i]['points'])



### Load in Toll et al. 2022 data
with open('POLLUTION_TRACKS/OCEAN_TRACKS/SHIP_TRACKS/SHIP.txt', 'r') as f:
    data = f.read().split('\n')[:-1]

# Get number of tracks at each time point
Toll_TrackNum = {}
i = 0
while i < len(data):
    if 'MYD' in data[i]:
            timestamp = data[i].split('MYD')[1].replace('.', '_')
            if timestamp in Toll_TrackNum.keys():
                 Toll_TrackNum[timestamp] += 1
            else:
                Toll_TrackNum[timestamp] = 1
    i += 1

# Format each track under timestamp and so forth
Toll_Tracks = {}
i = 0
j = 0
tracknum = 0
while i < len(data):
    if 'MYD' in data[i]:
            timestamp = data[i].split('MYD')[1].replace('.', '_')
            if timestamp in Toll_Tracks.keys():
                do_nothing = 'yes please'
                j += 1
            else:
                Toll_Tracks[timestamp] = [[] for i in range(Toll_TrackNum[timestamp])]
                j = 0
    else:
        Toll_Tracks[timestamp][j].append([float(data[i].strip().split('   ')[1]), float(data[i].strip().split('   ')[0])])
    i += 1

Toll_Centers = {}
for timestamp in Toll_Tracks.keys():
    centers = []
    for track in Toll_Tracks[timestamp]:
        #print(track)
        centers.append([np.mean(np.array(track)[:,0]), np.mean(np.array(track)[:,1])])
    Toll_Centers[timestamp] = centers





### Validate tracks against each other
# Validate Proposed Dataset against Song et al. 2022
validationSong = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])
for i in range(len(labelfiles)):
    print('File: {}'.format(i), end="\r")
    output = ValidationCheckSong(i, Song_Tracks, labelfiles, maskedfiles)
    validationSong = pd.concat([validationSong, output])
    validationSong['PointValid'] = validationSong['PointValid'].astype(bool)

# Validate Proposed Dataset against Toll et al. 2019
validationToll = pd.DataFrame(columns=['Timestamp','NumPierceTracks', 'NumTrueTracks','PierceTrack','PointValid','UnMatchedTrueTracks'])
for i in range(len(labelfiles)):
    print('File: {}'.format(i), end="\r")
    output = ValidationCheckToll(i, Toll_Tracks, labelfiles, maskedfiles)
    validationToll = pd.concat([validationToll, output])
    validationToll['PointValid'] = validationToll['PointValid'].astype(bool)

# Validate Proposed Dataset against Watson-Parris et al. 2022
validationWatson_Parris = ValidationCheckWatson_Parris(Watson_Parris_Tracks, maskedfiles)

# Validate Watson-Parris 2022 against Toll et al. 2019
validationWatsonParris_Toll = pd.DataFrame(columns=['Timestamp','NumTestTracks', 'NumTrueTracks','TestTrack','PointValid','UnMatchedTrueTracks'])
timestamps = list(Song_Tracks.keys())
for i in range(len(timestamps)):
    print('{}'.format(timestamps[i]), end="\r")
    output = ValidationCheckWatsonParris_Toll(timestamps[i], Toll_Centers, Watson_Parris_Tracks, labelfiles)
    validationWatsonParris_Toll = pd.concat([validationWatsonParris_Toll, output])
    validationWatsonParris_Toll['PointValid'] = validationWatsonParris_Toll['PointValid'].astype(bool)

# Validate Song et al 2019 against Watson-Parris 2022
validationSong_WatsonParris = ValidationCheckSong_WatsonParris(timestamp, Song_Tracks, Watson_Parris_Tracks)

# Validate Song et al 2019 against Toll et al 2019
validationSong_Toll = ValidationCheckSongAndToll(Song_Tracks, Toll_Tracks, Toll_Centers, labelfiles)

# Validate Watson-Parris et al. 2022 against Toll et al. 2019
validationWatsonParris_Toll = ValidationCheckWatsonParris_Toll(timestamp, Toll_Centers, Watson_Parris_Tracks, labelfiles)

# Calculate F1 scores for a given dataset

def F1Scores(validation_dataset):
    TP = validation_dataset['PointValid'].sum()
    FP = len(validation_dataset['PointValid']) - TP
    df = validation_dataset.groupby('Timestamp', as_index=False).first()
    FN = df['UnMatchedTrueTracks'].sum()
    tracknum = df['NumTestTracks'].sum() + df['NumTrueTracks'].sum() - TP

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    return(F1, Precision, Recall)


