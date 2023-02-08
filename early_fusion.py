import dask.dataframe as dd

from dask_ml.preprocessing import StandardScaler
import constants
from dask_ml.decomposition import PCA
import vaex
import vaex.ml
import numpy as np


def PCA_spectral():

    spectral = dd.read_csv(constants.AUDIO_BLF_SPECTRAL_PATH, sep='\t')
    y = spectral['id']
    spectral = spectral.drop(['id'], axis=1)

    spectral_arr = spectral.to_dask_array(lengths=True)
    scaler = StandardScaler()
    spectral_arr = scaler.fit_transform(X=spectral_arr)
    pca = PCA(n_components=80)
    res = pca.fit_transform(spectral_arr)
    print(pca.explained_variance_ratio_.cumsum())
    df = dd.from_dask_array(res)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/id_blf_spectral_mmsr_PCA.csv', sep='\t', single_file=True)


def PCA_incp():

    spectral = dd.read_csv(constants.VIDEO_INCP_PATH, sep='\t')
    y = spectral['id']
    spectral = spectral.drop(['id'], axis=1)

    spectral_arr = spectral.to_dask_array(lengths=True)
    scaler = StandardScaler()
    spectral_arr = scaler.fit_transform(X=spectral_arr)
    pca = PCA(n_components=819)
    res = pca.fit_transform(spectral_arr)
    print(pca.explained_variance_ratio_.cumsum())
    df = dd.from_dask_array(res)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/id_incp_mmsr_PCA.csv', sep='\t', single_file=True)


def PCA_resnet():

    spectral = dd.read_csv(constants.VIDEO_RESNET_PATH, sep='\t')
    y = spectral['id']
    spectral = spectral.drop(['id'], axis=1)

    spectral_arr = spectral.to_dask_array(lengths=True)
    scaler = StandardScaler()
    spectral_arr = scaler.fit_transform(X=spectral_arr)
    pca = PCA(n_components=1000)
    res = pca.fit_transform(spectral_arr)
    print(pca.explained_variance_ratio_.cumsum())
    df = dd.from_dask_array(res)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/id_resnet_mmsr_PCA.csv', sep='\t', single_file=True)


def PCA_logfluc():

    spectral = dd.read_csv(constants.AUDIO_BLF_LOGFLUC_PATH, sep='\t')
    y = spectral['id']
    spectral = spectral.drop(['id'], axis=1)
    spectral = spectral.drop(['ID'], axis=1)
    spectral = spectral.drop(['BLF_LOGFLUC3625'], axis = 1)

    spectral_arr = spectral.to_dask_array(lengths=True)
    scaler = StandardScaler()
    spectral_arr = scaler.fit_transform(X=spectral_arr)
    pca = PCA(n_components=360)
    res = pca.fit_transform(spectral_arr)
    print(pca.explained_variance_ratio_.cumsum())
    df = dd.from_dask_array(res)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/id_blf_logfluc_mmsr_PCA.csv', sep='\t', single_file=True)


def PCA_bert():

    spectral = dd.read_csv(constants.LYRICS_BERT_PATH, sep='\t')
    y = spectral['id']
    spectral = spectral.drop(['id'], axis=1)
    spectral_arr = spectral.to_dask_array(lengths=True)
    scaler = StandardScaler()
    spectral_arr = scaler.fit_transform(X=spectral_arr)
    pca = PCA(n_components=250)
    res = pca.fit_transform(spectral_arr)
    print(pca.explained_variance_ratio_.cumsum())
    df = dd.from_dask_array(res)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/id_lyrics_bert_mmsr_PCA.csv', sep='\t', single_file=True)



def PCA_vardeltaspectral():

    spectral = dd.read_csv(constants.AUDIO_BLF_VARDELTASPECTRAL_PATH, sep='\t')
    y = spectral['id']
    spectral = spectral.drop(['id'], axis=1)
    spectral_arr = spectral.to_dask_array(lengths=True)
    scaler = StandardScaler()
    spectral_arr = scaler.fit_transform(X=spectral_arr)
    pca = PCA(n_components=135)
    res = pca.fit_transform(spectral_arr)
    print(pca.explained_variance_ratio_.cumsum())
    df = dd.from_dask_array(res)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/id_blf_vardeltaspectral_mmsr_PCA.csv', sep='\t', single_file=True)



def merge_pre_PCAd_sets():
    """
    audio_VDS = dd.read_csv('data/id_blf_vardeltaspectral_mmsr_PCA.csv', sep = '\t')
    audio_VDS.repartition(npartitions=2)
    lyrics_bert = dd.read_csv('data/id_lyrics_bert_mmsr_PCA.csv', sep='\t')
    lyrics_bert.repartition(npartitions=2)
    #video_resnet = audio_VDS = dd.read_csv('data/id_resnet_mmsr_PCA.csv', sep = '\t')


    df = dd.merge(left=audio_VDS,right=lyrics_bert,on= 'id')
    df.head()

    """
    audio_VDS = vaex.read_csv('data/id_blf_vardeltaspectral_mmsr_PCA.csv', sep='\t')
    lyrics_bert = vaex.read_csv('data/id_lyrics_bert_mmsr_PCA.csv', sep='\t')
    video_incp =  vaex.read_csv('data/id_incp_mmsr_PCA.csv', sep='\t')
    for column_name in audio_VDS.column_names:
        new_column_name = f'{column_name}_a'
        audio_VDS.rename(column_name, new_column_name)

    audio_VDS.rename('id_a','id')

    for column_name in video_incp.column_names:
        new_column_name = f'{column_name}_incp'
        video_incp.rename(column_name, new_column_name)

    video_incp.rename('id_incp', 'id')





    joined =  audio_VDS.join(lyrics_bert,on = 'id')
    joined2 = joined.join(video_incp, on = 'id')
    print(joined2.head())
    joined2.export_csv('data/Pre_PCA_INCP_BERT_VDS.csv',sep = '\t')



def scale():
    file = dd.read_csv('data/Pre_PCA_INCP_BERT_VDS.csv', sep='\t')
    y = file['id']
    file = file.drop(['id'], axis=1)

    arr = file.to_dask_array(lengths=True)
    scaler = StandardScaler()
    arr = scaler.fit_transform(X=arr)
    df = dd.from_dask_array(arr)
    df = df.repartition(npartitions=200)
    df = df.reset_index(drop=True)
    y = y.repartition(npartitions=200)
    y = y.reset_index(drop=True)
    df = df.assign(id=y)
    print(df.head())
    dd.to_csv(df, 'data/Pre_PCA_INCP_BERT_VDS_SCALED.csv', sep='\t', single_file=True)






PCA_spectral()
#PCA_vardeltaspectral()
#PCA_logfluc()
#PCA_resnet()
#PCA_incp()
#PCA_bert()
#merge_pre_PCAd_sets()
#scale()