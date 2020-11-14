import { Component, OnInit,TemplateRef, ViewChild } from '@angular/core';
import {AnnotationsApiService} from '../annotation.service';
import {Subscription} from 'rxjs/Subscription';
import {Router, ActivatedRoute} from '@angular/router';
import { DomSanitizer  } from '@angular/platform-browser';
import {AnnotatedImage} from '../annotation.model'
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
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
  @ViewChild('successUploadModal', {static: false}) successUploadModal : TemplateRef<any>; // Note: TemplateRef
  constructor(private annotationsApi: AnnotationsApiService, 
              private sanitizer: DomSanitizer, 
              private router:Router, 
              private route: ActivatedRoute,
              private modalService: NgbModal) { }

  ngOnInit() {
    this.patientsSubscription = this.annotationsApi
                                              .get_sampled_images()
      .subscribe(res => {
                  this.imagesList = res.map((a: string)=> new AnnotatedImage(a));
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
    return this.sanitizer.bypassSecurityTrustStyle(`width: ${value}%`);
  }
  getCurrentClass()
  {
    var  currentClass = this.imagesList[this.currentImageIndex].url.indexOf("class0") == -1 ? "cancer":"pas cancer";
    return currentClass
  }
  toSubmit()
  {
    return  this.getTotalAnnotated() == this.imagesList.length;
  }
  reset()
  {
    this.router.routeReuseStrategy.shouldReuseRoute = () => false;
    this.router.onSameUrlNavigation = 'reload';
    this.router.navigate(['./'], { relativeTo: this.route });
  }
  submit()
  {
    this.annotationsApi.upload_annotated_data(this.imagesList).subscribe(res=> 
      {
        console.log(res)
        this.modalService.open(this.successUploadModal);
      });
  }
  closeModal(modal:any)
  {
    modal.close();
    this.reset()
  }
}
