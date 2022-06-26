import lightning as L
from lai_components.vsc_streamlit import StreamlitFrontend

import streamlit as st
import py3Dmol
from stmol import showmol

class My3D(L.LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  def run(self, *args, **kwargs):
    pass
  def configure_layout(self):
    return(StreamlitFrontend(render_fn = render))

def render(state):
  st.sidebar.title('Show Proteins')
  prot_str='1A2C,1BML,1D5M,1D5X,1D5Z,1D6E,1DEE,1E9F,1FC2,1FCC,1G4U,1GZS,1HE1,1HEZ,1HQR,1HXY,1IBX,1JBU,1JWM,1JWS'
  prot_list=prot_str.split(',')
  bcolor = st.sidebar.color_picker('Pick A Color', '#00f900')
  protein=st.sidebar.selectbox('select protein',prot_list)
  style = st.sidebar.selectbox('style',['line','cross','stick','sphere','cartoon','clicksphere'], index=3)
  spin = st.sidebar.checkbox('Spin', value = True)
  xyzview = py3Dmol.view(query='pdb:'+protein)
  xyzview.setStyle({style:{'color':'spectrum'}})
  xyzview.setBackgroundColor(bcolor)
  if spin:
      xyzview.spin(True)
  else:
      xyzview.spin(False)
  xyzview.zoomTo()
  showmol(xyzview,height=500,width=800)


