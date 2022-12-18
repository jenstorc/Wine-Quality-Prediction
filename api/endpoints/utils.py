
# from fastapi import APIRouter
# import ConfigParser
# Config = ConfigParser.ConfigParser()
# print(Config)
# #<ConfigParser.ConfigParser instance at 0x00BA9B20>
# Config.read("myfile.ini")
# #['c:\\tomorrow.ini']
# Config.sections()
# #['Others', 'SectionThree', 'SectionOne', 'SectionTwo']
# Config.options('SectionOne')
# #['Status', 'Name', 'Value', 'Age', 'Single']
# Config.get('SectionOne', 'Status')
# #'Single'
# from ../config/config.ini import TAG, VERSION, URL_DOC, URL_SWAGGER, CONTACTS

# router = APIRouter(tags=[TAG])

# @router.get("/")
# async def about() -> Dict[str, str]:
#     """Give informations about the API.

#     Returns:
#         Dict[str, str]: With shape :
#     - 
#     {"app_version": <VERSION>, "api_url_doc": <URL_DOC>, "api_url_swagger": <URL_SWAGGER>, "app_contacts": <CONTACTS>}
#     -
#     """

#     return {
#         "app_version": VERSION,
#         "api_url_doc": URL_DOC,
#         "api_url_swagger": URL_SWAGGER,
#         "app_contacts": CONTACTS
#     }

#     # sérialisé : stocker le resultat d'un model pour pouvoir le réustiliser 