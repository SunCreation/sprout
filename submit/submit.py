#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chromedriver_autoinstaller
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
import argparse as ap
import os


parser = ap.ArgumentParser()
# parser.add_argument('-c','--config',default='config.yaml',type=str,help='config yaml file')
parser.add_argument('-s','--submit_file',default='pre',type=str,help='file to submit')
parser.add_argument('-p','--post_dir',default='post',type=str,help='filepath to store')

args = parser.parse_args()

chromedriver_autoinstaller.install()
driver = webdriver.Chrome()


# In[2]:


driver.get('https://aifactory.space/competition/submission/2103')

sleep(0.2)
# In[10]:


_, email, passwd= driver.find_elements_by_css_selector('.ng-untouched.ng-pristine.ng-invalid')


# In[11]:

with open('submit/private') as f:
    email.send_keys(f.readline().strip())
    passwd.send_keys(f.readline().strip())


# In[13]:


driver.find_element_by_css_selector('.btn_b_u100_r5').click()


# In[17]:


#driver.find_element_by_css_selector('.file_btn').click()

sleep(2)
# In[18]:


from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
import os.path

# JavaScript: HTML5 File drop
# source            : https://gist.github.com/florentbr/0eff8b785e85e93ecc3ce500169bd676
# param1 WebElement : Drop area element
# param2 Double     : Optional - Drop offset x relative to the top/left corner of the drop area. Center if 0.
# param3 Double     : Optional - Drop offset y relative to the top/left corner of the drop area. Center if 0.
# return WebElement : File input
JS_DROP_FILES = "var k=arguments,d=k[0],g=k[1],c=k[2],m=d.ownerDocument||document;for(var e=0;;){var f=d.getBoundingClientRect(),b=f.left+(g||(f.width/2)),a=f.top+(c||(f.height/2)),h=m.elementFromPoint(b,a);if(h&&d.contains(h)){break}if(++e>1){var j=new Error('Element not interactable');j.code=15;throw j}d.scrollIntoView({behavior:'instant',block:'center',inline:'center'})}var l=m.createElement('INPUT');l.setAttribute('type','file');l.setAttribute('multiple','');l.setAttribute('style','position:fixed;z-index:2147483647;left:0;top:0;');l.onchange=function(q){l.parentElement.removeChild(l);q.stopPropagation();var r={constructor:DataTransfer,effectAllowed:'all',dropEffect:'none',types:['Files'],files:l.files,setData:function u(){},getData:function o(){},clearData:function s(){},setDragImage:function i(){}};if(window.DataTransferItemList){r.items=Object.setPrototypeOf(Array.prototype.map.call(l.files,function(x){return{constructor:DataTransferItem,kind:'file',type:x.type,getAsFile:function v(){return x},getAsString:function y(A){var z=new FileReader();z.onload=function(B){A(B.target.result)};z.readAsText(x)},webkitGetAsEntry:function w(){return{constructor:FileSystemFileEntry,name:x.name,fullPath:'/'+x.name,isFile:true,isDirectory:false,file:function z(A){A(x)}}}}}),{constructor:DataTransferItemList,add:function t(){},clear:function p(){},remove:function n(){}})}['dragenter','dragover','drop'].forEach(function(v){var w=m.createEvent('DragEvent');w.initMouseEvent(v,true,true,m.defaultView,0,0,0,b,a,false,false,false,false,0,null);Object.setPrototypeOf(w,null);w.dataTransfer=r;Object.setPrototypeOf(w,DragEvent.prototype);h.dispatchEvent(w)})};m.documentElement.appendChild(l);l.getBoundingClientRect();return l"

def drop_files(element, files, offsetX=0, offsetY=0):
    driver = element.parent
    isLocal = not driver._is_remote or '127.0.0.1' in driver.command_executor._url
    paths = []
    
    # ensure files are present, and upload to the remote server if session is remote
    for file in (files if isinstance(files, list) else [files]) :
        if not os.path.isfile(file) :
            raise FileNotFoundError(file)
        paths.append(file if isLocal else element._upload(file))
    
    value = '\n'.join(paths)
    elm_input = driver.execute_script(JS_DROP_FILES, element, offsetX, offsetY)
    elm_input._execute('sendKeysToElement', {'value': [value], 'text': value})

WebElement.drop_files = drop_files


# In[21]:


dropzone = driver.find_element_by_css_selector('.dropzone')
dropzone


# In[23]:

submit_file = args.submit_file
print(submit_file)
dropzone.drop_files(os.path.abspath(submit_file))


# In[24]:
sleep(0.2)

driver.find_element_by_css_selector('input.name').clear()
driver.find_element_by_css_selector('input.name').send_keys(submit_file.split('/')[-1])

sleep(0.2)

submit = [i for i in driver.find_elements_by_css_selector('.btn') if i.text=='업로드' ][0]
submit


# In[25]:


submit.click()

os.system(f'mv "{submit_file}" "{args.post_dir}"')

# In[27]:
sleep(1.5)

check=driver.find_element_by_css_selector('.btn_b_r35.ng-star-inserted')
check.click()

driver.close()

