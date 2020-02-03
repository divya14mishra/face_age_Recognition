
# using shutil module
import shutil
shutil.make_archive("face_captures_folder", "zip", "C:\\Users\\Hp\\PycharmProjects\\face_pictures")

# using zipfile module
import zipfile
my_zip = zipfile.ZipFile('first_zip.zip', 'w')
my_zip.write('C:\\Users\\Hp\\PycharmProjects\\face_pictures\\sample2.jpg')
my_zip.close()
