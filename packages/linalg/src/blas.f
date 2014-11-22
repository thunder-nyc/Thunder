c     Copyright 2014 Xiang Zhang All Rights Reserved.
c
c     Licensed under the Apache License, Version 2.0 (the "License");
c     you may not use this file except in compliance with the License.
c     You may obtain a copy of the License at
c
c          http://www.apache.org/licenses/LICENSE-2.0
c
c     Unless required by applicable law or agreed to in writing, software
c     distributed under the License is distributed on an "AS IS" BASIS,
c     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
c     See the License for the specific language governing permissions and
c     limitations under the License.
c
c     This file contains subroutine wrappers for BLAS functions returning
c     complex. The functions are appended with "subtd" which indicates
c     "subroutine for thunder".
c
      subroutine cdotcsubtd(r,n,x,incx,y,incy)
      external cdotc
      complex cdotc,r
      integer n,incx,incy
      complex x(*),y(*)
      dotc=cdotc(n,x,incx,y,incy)
      return
      end
c
      subroutine zdotcsubtd(r,n,x,incx,y,incy)
      external zdotc
      double complex zdotc,r
      integer n,incx,incy
      double complex x(*),y(*)
      dotc=zdotc(n,x,incx,y,incy)
      return
      end
c
      subroutine cdotusubtd(r,n,x,incx,y,incy)
      external cdotu
      complex cdotu,r
      integer n,incx,incy
      complex x(*),y(*)
      dotu=cdotu(n,x,incx,y,incy)
      return
      end
c
      subroutine zdotusubtd(r,n,x,incx,y,incy)
      external zdotu
      double complex zdotu,r
      integer n,incx,incy
      double complex x(*),y(*)
      dotu=zdotu(n,x,incx,y,incy)
      return
      end
