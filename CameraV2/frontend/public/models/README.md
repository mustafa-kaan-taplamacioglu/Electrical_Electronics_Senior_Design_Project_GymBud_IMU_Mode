# 3D Avatar Modelleri

Bu klasöre 3D avatar modellerinizi koyun.

## Ready Player Me Avatar Oluşturma

1. https://readyplayer.me/ sitesine gidin
2. Avatar oluşturun (ücretsiz)
3. "Download" butonuna tıklayın
4. GLB formatını seçin
5. Dosyayı buraya `avatar.glb` olarak kaydedin

## Mixamo Karakter İndirme

1. https://www.mixamo.com/ sitesine gidin (Adobe hesabı gerekli - ücretsiz)
2. "Characters" bölümünden bir karakter seçin
3. "Download" butonuna tıklayın
4. Format: FBX Binary (.fbx) veya GLB seçin
5. Dosyayı buraya kaydedin

## Kullanım

Avatar modelini kullanmak için WorkoutSession.tsx'te:

```tsx
<HumanAvatar 
  landmarks={currentLandmarks} 
  width={400} 
  height={480}
  modelUrl="/models/avatar.glb"  // Model dosyanızın yolu
/>
```

## Desteklenen Formatlar

- `.glb` (tercih edilen)
- `.gltf`

