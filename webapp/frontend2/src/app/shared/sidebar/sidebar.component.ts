import { Component, OnInit } from '@angular/core';
import { AuthenticationService } from '../../_services/authentication.service';
import {User} from '../../_models/user';
@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent implements OnInit {

  public parentId = "";
  clickedMenu(event) {
    var target = event.currentTarget;
    let parentId = target.id;
    if (parentId == this.parentId) {
      console.log('same');
      this.parentId = "";
    } else {
      console.log('not same');
      this.parentId = target.id;
    }
  }

  currentUser: User;
  constructor(private authenticationService: AuthenticationService)
  {
    this.authenticationService.currentUser.subscribe(x =>
      {
        this.currentUser = x
      } );
  }

  ngOnInit() {
    const body = document.querySelector('body');

    // add class 'hover-open' to sidebar navitem while hover in sidebar-icon-only menu
    document.querySelectorAll('.sidebar .nav-item').forEach(function (el) {
      el.addEventListener('mouseover', function () {
        if (body.classList.contains('sidebar-icon-only')) {
          el.classList.add('hover-open');
        }
      });
      el.addEventListener('mouseout', function () {
        if (body.classList.contains('sidebar-icon-only')) {
          el.classList.remove('hover-open');
        }
      });
    });
  }

  isAnnotator()
  {
      if (this.currentUser == null)
          return false;
      if (this.currentUser.roles == null)
          return false;
      if (this.currentUser.roles.includes("annotator"))
          return true;
      return false;
  }
}
