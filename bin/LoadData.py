import mne
import os
import glob
import numpy as np

class LoadData:
    def __init__(self,eeg_file_path: str):
        self.eeg_file_path = eeg_file_path

    def load_raw_data_gdf(self,file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self,file_path_extension: str =''):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)

class LoadBCIC(LoadData):
    '''Subclass of LoadData for loading BCI Competition IV Dataset 2a'''
    def __init__(self, file_to_load,*args):
        self.stimcodes=('769','770','771','772')
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC,self).__init__(*args)

    def get_epochs(self, tmin=-4.5,tmax=5.0,baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data,
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data

class LoadKU(LoadData):
    '''Subclass of LoadData for loading KU Dataset'''
    def __init__(self,subject_id,*args):
        self.subject_id=subject_id
        self.fs=1000
        super(LoadKU,self).__init__(*args)

    def get_epochs(self,sessions=[1, 2]):
        for i in sessions:
            file_to_load=f'session{str(i)}/s{str(self.subject_id)}/EEG_MI.mat'
            self.load_raw_data_mat(file_to_load)
            x_data = self.raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
            x_data = np.transpose(x_data,axes=[1, 2, 0])
            labels = self.raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
            y_labels = labels - np.min(labels)
            if hasattr(self,'x_data'):
                self.x_data=np.append(self.x_data,x_data,axis=0)
                self.y_labels=np.append(self.y_labels,y_labels)
            else:
                self.x_data = x_data
                self.y_labels = y_labels
        ch_names = self.raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]
        eeg_data = {'x_data': self.x_data,
                    'y_labels': self.y_labels,
                    'fs': self.fs,
                    'ch_names':ch_names_list}

        return eeg_data
