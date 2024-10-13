import requests
import queue
from bs4 import BeautifulSoup, Tag, Comment, NavigableString
from symreg.formula import BinaryOp, UnaryOp
import threading, time
import re

CRAWL = True

visited = set()
visited.add("https://en.wikipedia.org/wiki/Main_Page")

queue = queue.Queue()
queue.put("https://en.wikipedia.org/wiki/Diffusion")
queue.put("https://en.wikipedia.org/wiki/Gravity")
queue.put("https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion")
queue.put("https://en.wikipedia.org/wiki/Laws_of_thermodynamics")
queue.put("https://en.wikipedia.org/wiki/Entropy")
queue.put("https://en.wikipedia.org/wiki/Mechanical_engineering")
queue.put("https://en.wikipedia.org/wiki/Turbulence")

unary_op_count = 0
binary_op_count = 0
op_count = {}
for op in UnaryOp:
  op_count[op] = 0
for op in BinaryOp:
  op_count[op] = 0

def count_op(op):
  global unary_op_count
  global binary_op_count
  global op_count

  if isinstance(op, UnaryOp):
    unary_op_count += 1
    op_count[op] += 1
  elif isinstance(op, BinaryOp):
    binary_op_count += 1
    op_count[op] += 1
  else:
    raise TypeError

def is_not_whitespace(elem):
  if isinstance(elem, Comment):
    return False
  if isinstance(elem, NavigableString):
    return elem.text != '\\n'
  if isinstance(elem, Tag):
    return elem.name != 'mspace'
  return True

def find_implicit_multiplications(elem):
  if not isinstance(elem, Tag):
    return

  def is_expr(elem):
    if isinstance(elem, Tag) and elem.name == 'mrow':
      if len(elem.contents) == 3 and elem.contents[1].name == 'mo':
        return False

    return elem.name in ["mi", "mn", "mrow", "msqrt"]

  contents = list(filter(is_not_whitespace, elem.contents))
  for i in range(len(contents) - 1):
    if not is_expr(contents[i]) or not is_expr(contents[i + 1]):
      continue

    # Try to ignore derivation operators
    if contents[i].text in ["d", "∂"]:
      continue

    count_op(BinaryOp.MUL)

  for child in contents:
    find_implicit_multiplications(child)

def parse_mathml_element(elem):
  for op in elem.find_all('mo'):
    if op.text == '+':
      count_op(BinaryOp.ADD)
    elif op.text == '-' or op.text == '−':
      count_op(BinaryOp.SUB)

  find_implicit_multiplications(elem)

  for _ in elem.find_all('mfrac'):
    count_op(BinaryOp.DIV)

  for _ in elem.find_all('msqrt'):
    count_op(UnaryOp.SQRT)

  for _ in elem.find_all('msup'):
    count_op(BinaryOp.POW)

  for ident in elem.find_all('mi'):
    if ident.text == 'sin':
      count_op(UnaryOp.SIN)
    elif ident.text == 'cos':
      count_op(UnaryOp.COS)
    elif ident.text == 'tan':
      count_op(UnaryOp.TAN)
    elif ident.text == 'asin' or ident.text == 'arcsin':
      count_op(UnaryOp.ASIN)
    elif ident.text == 'acos' or ident.text == 'arccos':
      count_op(UnaryOp.ACOS)
    elif ident.text == 'atan' or ident.text == 'arctan':
      count_op(UnaryOp.ATAN)
    elif ident.text == 'exp':
      count_op(UnaryOp.EXP)

def crawl(url):
  page = requests.get(url)
  content = re.sub(r'\s{2,}|[\n\r]', '', str(page.content))
  html = BeautifulSoup(content, 'html.parser')

  maths = html.find_all('span', attrs={'class': 'mwe-math-mathml-inline'})
  for math in maths:
    parse_mathml_element(math.contents[0])

  if len(maths) == 0:
    return

  if CRAWL:
    for link in html.find_all('a'):
      href = link.get('href')
      if not isinstance(href, str):
        continue
      if not href.startswith("/wiki/"):
        continue
      if "Special:" in href:
        continue
      if "Talk:" in href:
        continue
      if "Help:" in href:
        continue
      if "Wikipedia:" in href:
        continue
      if "Portal:" in href:
        continue
      if "File:" in href:
        continue
      if "Template:" in href:
        continue
      if "Category:" in href:
        continue

      new_url = f"https://en.wikipedia.org{href}"
      if new_url not in visited:
        queue.put(new_url)
        visited.add(new_url)

class CrawlerThread(threading.Thread):
  def __init__(self, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
    super().__init__(group, target, name, args, kwargs, daemon=daemon)
    self.should_exit = False

  def run(self):
    while not self.should_exit:
      try:
        url = queue.get_nowait()
        print(f"Crawl: {url}")
        crawl(url)
      except:
        continue

threads = [ CrawlerThread() for _ in range(4) ]
for thread in threads:
  thread.start()

should_exit = False
while not should_exit:
  try:
    time.sleep(1)
  except KeyboardInterrupt:
    for thread in threads:
      thread.should_exit = True

    for thread in threads:
      thread.join()

    should_exit = True

if unary_op_count > 0:
  print("Unary operators:")
  for op in UnaryOp:
    print(f"P({op}) = {(op_count[op]/unary_op_count)*100}%")

if binary_op_count > 0:
  print("Binary operators:")
  for op in BinaryOp:
    print(f"P({op}) = {(op_count[op]/binary_op_count)*100}%")
