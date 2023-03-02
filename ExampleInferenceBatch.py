import vtk
import os
import DataProcessing
import SimpleITK as sitk
import ModelConfiguration

# process example CT image

def Segment(image_path, filename, save_directory):
    isExist = os.path.exists(save_directory)
    if not isExist:
        os.makedirs(save_directory)

    image_name = os.path.join(save_directory, filename.split("_")[0])
    ctImage = sitk.ReadImage(image_path)
    binaryImage = DataProcessing.CreateBoneMask(ctImage)
    ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)
    #sitk.WriteImage(ctImage, "test/54/54MRcropToCT_toModel.mha")

    # model
    modelPath = './MiccaiFinalModel.dat'
    device = ModelConfiguration.getDevice()
    model = ModelConfiguration.adaptModel(modelPath, device)
    imageData = ModelConfiguration.adaptData(ctImage, device)

    landmarks, all_seven_labels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)

    sitk.WriteImage(all_seven_labels, image_name + "_labeled.nii.gz")

    point_writer = vtk.vtkXMLPolyDataWriter()
    point_writer.SetFileName(image_name + "_CTLPoints.vtp")
    point_writer.SetInputData(landmarks)
    point_writer.Write()


if __name__ == '__main__':
    data_dir = "D:\\Data\\CNH_Paired\\NoBedCTs"
    save_dir = "D:\\Data\\CNH_Paired\\nbCTsegs"
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        try:
            Segment(filepath, filename, save_dir)
        except Exception as e:
            print(f"For case {filename}, an error occurred:", e)
            continue
