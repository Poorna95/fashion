const downloader = require("./insta_photo_downloader").downloader;
const urlScraper = require("./insta_url_scrapper").urlScrapper;

//absolute path to the main directory (ex:- "C:/Users/User/Desktop/insta_photo_downloader")
const dirAbsPath = "C:/Users/Poorna/Desktop/insta_photo";

//output file location
const urlList = "./urlList.txt";

//facebook email (ex:- "example@email.com")
const fb_email = "poornapanduwawala077@gmail.com";

//facebook pass (ex:- "password")
const fb_pass = "po01or08na";

//maximun amount of images per username
const maxImagePerUsername = 30;

//username list ----- add new usernames -----
const usernames = ["dinakshie","asekawijewardena"];

(async () => {
  await urlScraper(
    usernames,
    maxImagePerUsername,
    urlList,
    fb_email,
    fb_pass,
    dirAbsPath
  );
  console.log("url scrape done");
  await downloader(urlList);
  console.log("all done");
})();
