---
toc: false
---

<div class="profile-header">
  <img src="/images/of-me/wfu-bme/headshot.jpg" alt="Adrien Babet Photo" class="headshot-photo" />
  <div class="profile-info">
    {{< hextra/hero-headline >}}Adrien Babet{{< /hextra/hero-headline >}}
    {{< hextra/hero-subtitle >}}I love biomechanics, programming, and snowboarding{{< /hextra/hero-subtitle >}}
  </div>
</div>

## About
I’m currently studying Kinesiology at Montana State University in Bozeman MT. Here, I’ve worked as an assistant in the Anatomy & Physiology cadaver lab, and conducted research in the Neuromuscular Biomechanics Laboratory. Outside of the classroom, I am vice-president of the Montana State University Nogi Jiu-Jitsu Club and work as a snowboard instructor during the winter. Whenever I can, I love coding in python and learning new sports. This website is intended as a learning tool for myself as well as to share my projects with others.

## Featured Content

{{< cards cols="2" >}}
  {{< card link="/projects/websat/" title="WebSAT" image="/images/of-me/wfu-bme/AdrienBabet1.jpg" subtitle="In the summer of 2025, I worked with Wake Forest University School of Medicine Center for Injury Biomechanics under Dr. Scott Gayzik and in collaboration with Elemance on a novel web based signal analysis tool for vehicle safety testing." >}}
  {{< card link="/projects/kalman/" title="Kalman Filtering" image="images/imu-bryan.jpeg" subtitle="IMU's are an increasingly popular way to gather movement data as wearable technology improves. However, the accelerometers and gyroscopes within them are noisy, making signal filters necessary to decrease error." method="Fill" options="3060x2040" >}}
{{< /cards >}}

## More

{{< cards >}}
  {{< card link="/projects/" title="Content" icon="collection">}}
  {{< card link="https://adrienbabet.com/pdfs/CV.pdf" title="CV" icon="document-text">}}
  {{< card link="https://www.linkedin.com/in/adrien-babet-37bb29301/" title="LinkedIn" icon="linkedin">}}
{{< /cards >}}

<style>
  .headshot-photo {
    box-shadow: 0 4px 5px rgba(0, 0, 0, 0.15);
  }
  .dark .headshot-photo {
    box-shadow: 0 4px 5px rgba(255, 255, 255, 0.15);
  }

  .profile-header {
    display: flex;
    flex-direction: row;
    gap: 2rem;
    justify-content: flex-start;
    align-items: center;
    margin: 0;
    padding: 0;
  }
  .headshot-photo {
    width: 9rem;
    height: 9rem;
    border-radius: 9999px;
    object-fit: cover;
    margin: 0;
    padding: 0;
    box-shadow: 0 4px 5px rgba(0, 0, 0, 0.15);
  }
  @media (min-width: 768px) {
    .headshot-photo {
      width: 11rem;
      height: 11rem;
    }
  }
  @media (min-width: 1024px) {
    .headshot-photo {
      width: 12rem;
      height: 12rem;
    }
  }
  .dark .headshot-photo {
    box-shadow: 0 4px 5px rgba(255, 255, 255, 0.15);
  }
  .profile-info {
    display: flex;
    flex-direction: column;
    margin: 0;
    padding: 0;
  }
  .profile-info h1 {
    margin: 0 0 0 0;
    padding: 0;
  }
  .profile-info p {
    margin: 0;
    padding: 0;
  }
</style>

*Credit: This site uses the [Hextra](https://github.com/imfing/hextra) theme.*
