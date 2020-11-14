import { Component, OnInit } from '@angular/core';
import {AnnotationsApiService} from '../annotation.service';
import {Subscription} from 'rxjs/Subscription';
import { DomSanitizer  } from '@angular/platform-browser';
import {AnnotatedImage} from '../annotation.model'
@Component({
  selector: 'app-annotate',
  templateUrl: './annotate.component.html',
  styleUrls: ['./annotate.component.scss']
})
export class AnnotateComponent implements OnInit {

  patientsSubscription: Subscription;
  imagesList : AnnotatedImage[];
  currentImage : AnnotatedImage;
  currentImageIndex : number = 0;
  constructor(private annotationsApi: AnnotationsApiService, private sanitizer: DomSanitizer) { }

  ngOnInit() {
    this.patientsSubscription = this.annotationsApi
                                              .get_sampled_images()
      .subscribe(res => {
                  this.imagesList = res.map((a: string)=> new AnnotatedImage(a));;
                  this.currentImage = this.imagesList[0]
        },
        console.error
      );
  }

  hasPrev()
  {
    return this.currentImageIndex > 0
  }

  onPrev()
  {
    this.currentImageIndex--
  }
  onNotCancer()
  {
    this.imagesList[this.currentImageIndex].hasCancer = false;
    if (this.hasNext())
        this.onNext();
  }
  onCancer()
  {
    this.imagesList[this.currentImageIndex].hasCancer = true;
    if (this.hasNext())
        this.onNext();
  }
  hasNext()
  {
    return this.currentImageIndex  < this.imagesList.length - 1;
  }
  onNext()
  {
    this.currentImageIndex++
  }

  getTotalAnnotated()
  {
    return this.imagesList.filter(a=> a.hasCancer != undefined).length
  }
  getWidthStyle()
  {
    var value = (this.getTotalAnnotated() * 100)/this.imagesList.length;
    console.log(value)
    return this.sanitizer.bypassSecurityTrustStyle(`width: ${value}%`);
  }
  getCurrentFileName()
  {
    return this.imagesList[this.currentImageIndex].url.replace(/^.*[\\\/]/, '')
  }
}
