import requests
import bs4


def get_soup(domain, page):
    response = requests.get(domain+page)
    return bs4.BeautifulSoup(response.text, "html.parser")


domain = 'https://icml.cc'
pages = ['/Conferences/2019/ScheduleMultitrack?text=&session=&day=2019-06-11&event_type=',
         '/Conferences/2019/ScheduleMultitrack?session=&event_type=&day=2019-06-12',
         '/Conferences/2019/ScheduleMultitrack?session=&event_type=&day=2019-06-13'
         ]
for page in pages:
    soup = get_soup(domain, page)
    i = 0
    for a in soup.find_all(attrs={'class': 'eventlink'}, href=True):
        i += 1
        msg = str(i)+". " + a.string + " [[Web]("+str(domain+a['href'])+")] "
        sub_soup = get_soup(domain, a['href'])
        for slides in sub_soup.find_all(attrs={'class': 'btn btn-default btn-xs href_PDF'}, href=True):
            msg = msg + "[[Slides]("+str(domain+slides['href'])+")]"
        print(msg)
