import pandas as pd
import os

def generate_mias_csv():
    # 1. Definir rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_path = os.path.join(project_root, 'data', 'info_mias.csv')
    os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)

    # 2. Base de datos completa (Lógica de generación para las 322 imágenes)
    # Lista de IDs que tienen anomalías y su severidad (B=1, M=2)
    # Formato: (ID, Severidad)
    abnormalities = {
        1:1, 2:1, 5:1, 10:1, 12:1, 13:1, 15:1, 17:1, 19:1, 21:1, 23:2, 25:1, 28:2, 30:1, 32:1, 
        58:2, 59:1, 63:1, 69:1, 72:2, 75:2, 80:1, 81:1, 83:1, 90:2, 91:1, 92:2, 95:2, 97:1, 
        99:1, 102:2, 104:1, 105:2, 107:1, 110:2, 111:2, 115:2, 117:2, 120:2, 121:1, 124:2, 
        125:2, 126:1, 127:1, 130:2, 132:1, 134:2, 141:2, 142:1, 144:2, 145:1, 148:2, 150:1, 
        152:1, 155:2, 158:2, 160:1, 163:1, 165:1, 167:1, 170:2, 171:2, 175:1, 178:2, 179:2, 
        181:2, 184:2, 186:2, 188:2, 190:1, 191:1, 193:1, 195:1, 198:1, 199:1, 202:2, 204:1, 
        206:2, 207:1, 209:2, 211:2, 212:1, 213:2, 214:1, 216:2, 218:1, 219:1, 222:1, 223:1, 
        226:1, 227:1, 231:2, 233:2, 236:1, 238:2, 239:2, 240:1, 241:2, 244:1, 245:2, 248:1, 
        249:2, 252:1, 253:2, 256:2, 264:2, 265:2, 267:2, 270:2, 271:2, 274:2, 290:1, 312:1, 
        314:1, 315:1
    }

    rows = []
    # MIAS tiene imágenes desde la 1 hasta la 322
    for i in range(1, 323):
        img_id = f"mdb{i:03}" # Genera mdb001, mdb002...
        
        if i in abnormalities:
            target = abnormalities[i]
            severity = 'B' if target == 1 else 'M'
            class_name = 'ABNORM'
        else:
            target = 0
            severity = 'N'
            class_name = 'NORM'
            
        rows.append({
            'file_name': img_id,
            'tissue_type': 'UNK', # Simplificado ya que no afecta a la clasificación
            'class': class_name,
            'severity': severity,
            'target': target
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"✅ ¡PROYECTO SALVADO!")
    print(f"Archivo generado en: {output_path}")
    print(f"Total de registros: {len(df)}")
    print("\nDistribución de clases:")
    print(df['target'].value_counts().rename(index={0:'Normal', 1:'Benigno', 2:'Maligno'}))

if __name__ == "__main__":
    generate_mias_csv()